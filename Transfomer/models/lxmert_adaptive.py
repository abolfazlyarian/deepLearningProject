import math
import os
import sys
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertConfig, BertTokenizer
# from transformers.modeling_bert import BertLayerNorm

from .adaptive_span import AdaptiveSpan
from .entmax import EntmaxAlpha
from .layerdrop import LayerDrop_Bert, LayerDrop_Cross
from .lxmert_utils import (VISUAL_CONFIG, BertPreTrainedModel, InputFeatures,
                           convert_sents_to_features, set_visual_config)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MAX_VQA_LENGTH = 20
bert_config = BertConfig()


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)


## BertEmbeddings
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=0
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size, padding_idx=0
        )
        
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


## BertAttention


class BertAttention(nn.Module):
    """
    from transformers import BertConfig
    
    bert_att = BertAttention(BertConfig())
    context_output = bert_att(hidden_states = torch.rand(128,20,768),
                          context = torch.rand(128,36,768),
                          attention_mask = None)
    context_output.shape # [128, 20, 768]

    """

    def __init__(self, config, params):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = (
            config.num_attention_heads
        )  # params['num_attention_heads'] # 12
        self.attention_head_size = (
            config.hidden_size // config.num_attention_heads
        )  # 768/12
        self.all_head_size = (
            self.num_attention_heads * self.attention_head_size
        )  # 12*64

        self.query = nn.Linear(config.hidden_size, self.all_head_size)  # 768x768
        self.key = nn.Linear(config.hidden_size, self.all_head_size)  # 768x768
        self.value = nn.Linear(config.hidden_size, self.all_head_size)  # 768x768

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.adapt_span_bool = params["adapt_span_enabled"]
        self.sparse = params["sparse_enabled"]

        if self.sparse:
            self.entmax_alpha = EntmaxAlpha(self.num_attention_heads)

        if self.adapt_span_bool:
            self.adaptive_span = AdaptiveSpan(
                params["adapt_span_enabled"],
                params["attn_span"],
                params["adapt_span_loss_coeff"],
                params["adapt_span_ramp"],
                params["adapt_span_init"],
                params["adapt_span_cache"],
                params["nb_heads"],
                params["bs"],
                params["mask_size"],
            )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        # print('Hidden States: ', hidden_states.shape)
        # print('context: :', context.shape)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.sparse:
            attention_probs = self.entmax_alpha(attention_scores)
        else:
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.adapt_span_bool:
            attention_probs = self.adaptive_span(attention_probs)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

    def get_cache_size(self):
        return self.adaptive_span.get_cache_size()


class BertAttOutput(nn.Module):
    """
    from transformers import BertConfig
    bert_att_output = BertAttOutput(BertConfig())
    output = bert_att_output(torch.rand(128,20,768),torch.rand(128,20,768))
    output.shape [128,20,768]

    """

    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


## BertCross Attention
class BertCrossattLayer(nn.Module):
    """
    from transformers import BertConfig
    
    bert_cross_att = BertCrossattLayer(BertConfig())
    output = bert_cross_att(input_tensor = torch.rand(128,20,768), 
                        ctx_tensor = torch.rand(128,36,768), 
                        ctx_att_mask = None)
                        
    output.shape [128,20,768]
    """

    def __init__(self, config, params):
        super().__init__()
        self.att = BertAttention(config, params=params)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)  # [128,20,768]
        attention_output = self.output(output, input_tensor)
        return attention_output


class BertSelfattLayer(nn.Module):
    """
    bert_self_att_layer = BertSelfattLayer(bert_config)
    output = bert_self_att_layer(input_tensor = torch.rand(128,20,768),
                             attention_mask = torch.rand(128,1,1,20))
    output.shape [128, 20, 768]
    """

    def __init__(self, config, params):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config, params=params)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    """
    bert_intermediate = BertIntermediate(bert_config)
    output = bert_intermediate(torch.rand(128,20,768))
    output.shape # [128,20,3072]

    """

    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (
            sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.intermediate_act_fn = GeLU()
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """
    bert_output = BertOutput(bert_config)
    output = bert_output(hidden_states = torch.rand(128,20,3072),
                         input_tensor = torch.rand(128,20,768))
    output.shape # [128,20,768]

    """

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(
            config.intermediate_size, config.hidden_size
        )  # [3072x768]
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """
    from transformers import BertConfig
    bert_layer  = BertLayer(BertConfig())
    output = bert_layer(torch.rand(128,20,768),torch.rand(128,1,1,20))
    output.shape [128,20,768]
    """

    def __init__(self, config, params):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config, params)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(
            hidden_states, attention_mask
        )  # [128, 20, 768]
        intermediate_output = self.intermediate(attention_output)
        # [128,20,3072], [128,20,768]
        layer_output = self.output(
            intermediate_output, attention_output
        )  # [128,20,768]
        return layer_output


class LXRTXLayer(nn.Module):
    """
    from transformers import BertConfig
    lxrtx_layer = LXRTXLayer(BertConfig())
    output = lxrtx_layer(lang_feats = torch.rand(128,20,768),
                      lang_attention_mask = torch.rand(128,1,1,20),
                      visn_feats = torch.rand(128,36,768),
                      visn_attention_mask = None)

    lang_output.shape: [128,20,768]
    visn_output.shape: [128,36,768]
    
    """

    def __init__(self, config, params):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(config, params)

        # Self-attention Layers
        self.lang_self_att = BertSelfattLayer(config, params)
        self.visn_self_att = BertSelfattLayer(config, params)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def cross_att(
        self, lang_input, lang_attention_mask, visn_input, visn_attention_mask
    ):
        # Cross Attention
        lang_att_output = self.visual_attention(
            lang_input, visn_input, ctx_att_mask=visn_attention_mask
        )
        visn_att_output = self.visual_attention(
            visn_input, lang_input, ctx_att_mask=lang_attention_mask
        )
        return lang_att_output, visn_att_output

    def self_att(
        self, lang_input, lang_attention_mask, visn_input, visn_attention_mask
    ):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_att_output, visn_att_output = self.cross_att(
            lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask
        )

        lang_att_output, visn_att_output = self.self_att(
            lang_att_output, lang_attention_mask, visn_att_output, visn_attention_mask
        )
        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output)

        return lang_output, visn_output


class VisualFeatEncoder(nn.Module):
    """

    from transformers import BertConfig
    
    visual_feat_encoder = VisualFeatEncoder(BertConfig())
    output = visual_feat_encoder((torch.rand(128,36,2048),torch.rand(128,36,4))) img_feats+box_feats
    
    output.shape: [128,36,768]
    """

    def __init__(self, config):
        super().__init__()
        feat_dim = VISUAL_CONFIG.visual_feat_dim
        pos_dim = VISUAL_CONFIG.visual_pos_dim

        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)

        # Box position encoding
        self.box_fc = nn.Linear(pos_dim, config.hidden_size)
        self.box_layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visn_input):
        feats, boxes = visn_input

        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        y = self.box_fc(boxes)
        y = self.box_layer_norm(y)
        output = (x + y) / 2

        output = self.dropout(output)
        return output


class LXRTEncoder(nn.Module):
    """
    from transformers import BertConfig
    lxrt_encoder = LXRTEncoder(BertConfig())

    output = lxrt_encoder(lang_feats = torch.rand(128,20,768),
                      lang_attention_mask = torch.rand(128,1,1,20),
                      visn_feats = (torch.rand(128,36,2048),torch.rand(128,36,4)),
                      visn_attention_mask = None)

    lang_feats.shape: [128,20,768]
    visn_feats.shape: [128,36,768]

    """

    def __init__(self, config, params):
        super().__init__()

        # Obj-level image embedding layer
        self.params = params
        self.visn_fc = VisualFeatEncoder(config)

        # Number of layers
        self.num_l_layers = VISUAL_CONFIG.l_layers
        self.num_x_layers = VISUAL_CONFIG.x_layers
        self.num_r_layers = VISUAL_CONFIG.r_layers
        print(
            "LXRT encoder with %d l_layers, %d x_layers, and %d r_layers."
            % (self.num_l_layers, self.num_x_layers, self.num_r_layers)
        )

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.ModuleList(
            [BertLayer(config, params=params) for _ in range(self.num_l_layers)]
        )
        self.r_layers = nn.ModuleList(
            [BertLayer(config, params=params) for _ in range(self.num_r_layers)]
        )
        self.x_layers = nn.ModuleList(
            [LXRTXLayer(config, params=params) for _ in range(self.num_x_layers)]
        )

        if self.params["layerdrop_enabled"] == True:
            self.layer = LayerDrop_Bert(self.layer, self.params["layerdrop_num_layers"])
            self.r_layers = LayerDrop_Bert(
                self.r_layers, self.params["layerdrop_num_layers"]
            )
            self.x_layers = LayerDrop_Cross(
                self.x_layers, self.params["layerdrop_num_layers"]
            )

    def forward(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask=None
    ):
        # Run visual embedding layer
        # Note: Word embedding layer was executed outside this module.
        #       Keep this design to allow loading BERT weights.
        visn_feats = self.visn_fc(visn_feats)  #  [bs, 36, 768]

        if not self.params["layerdrop_enabled"]:
            # Run language layers
            for layer_module in self.layer:
                lang_feats = layer_module(lang_feats, lang_attention_mask)
            # Run relational layers
            for layer_module in self.r_layers:
                visn_feats = layer_module(visn_feats, visn_attention_mask)
            # Run cross-modality layers
            for layer_module in self.x_layers:
                lang_feats, visn_feats = layer_module(
                    lang_feats, lang_attention_mask, visn_feats, visn_attention_mask
                )
        else:
            lang_feats = self.layer(lang_feats, lang_attention_mask)
            visn_feats = self.r_layers(visn_feats, visn_attention_mask)
            lang_feats, visn_feats = self.x_layers(
                lang_feats, lang_attention_mask, visn_feats, visn_attention_mask
            )

        return lang_feats, visn_feats


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LXRTModel(BertPreTrainedModel):
    """
    LXRT Model.
    
    model = LXRTModel.from_pretrained("bert-base-uncased")
    
    output = model(input_ids = torch.rand(128,20).long(), 
               token_type_ids = torch.rand(128,20).long(),
               attention_mask = torch.rand(128,32).long(),
               visual_feats = (torch.rand(128,36,2048),torch.rand(128,36,4)),
               visual_attention_mask = None)
    
    
    lang_feats.shape -> [128, 20, 768]
    vision_feats.shape -> [128, 36, 768]
    pooled_output.shape -> [128,768]
    
    """

    def __init__(self, config, params):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = LXRTEncoder(config, params)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        visual_feats=None,
        visual_attention_mask=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.

        # print('Attention Mask', attention_mask.shape) : [128, 20]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # print('Extended Attention Mask', extended_attention_mask.shape): [128, 1, 1, 20]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # print('Extended Attention Mask 1k', extended_attention_mask.shape): [128, 1, 1, 20]

        # Process the visual attention mask
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(
                1
            ).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            extended_visual_attention_mask = (
                1.0 - extended_visual_attention_mask
            ) * -10000.0
        else:
            extended_visual_attention_mask = None

        # print('Extended Visual Attention Mask', extended_visual_attention_mask.shape) Shape: None

        # Positional Word Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # print('Embedding Output', embedding_output.shape): [128,20,768]

        # Run LXRT backbone

        lang_feats, visn_feats = self.encoder(
            embedding_output,
            extended_attention_mask,
            visn_feats=visual_feats,
            visn_attention_mask=extended_visual_attention_mask,
        )

        pooled_output = self.pooler(lang_feats)

        return (lang_feats, visn_feats), pooled_output


class VisualBertForLXRFeature(BertPreTrainedModel):
    """
    BERT model for classification.
    
    bert = VisualBertForLXRFeature.from_pretrained("bert-base-uncased",mode='x')
    
    output = bert(input_ids = torch.rand(128,20).long(), 
              token_type_ids = torch.rand(128,20).long(),
              attention_mask = torch.rand(128,20).long(),
              visual_feats = (torch.rand(128,36,2048),torch.rand(128,36,4)), # for feats and boxes
              visual_attention_mask = None)
              
    output.shape -> [128,768]          ,
    """

    def __init__(
        self, config, params, mode="lxr",
    ):
        """
        :param config:
        :param mode:  Number of visual layers
        """
        super().__init__(config)
        self.bert = LXRTModel(config, params)
        self.mode = mode
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        visual_feats=None,
        visual_attention_mask=None,
    ):
        feat_seq, pooled_output = self.bert(
            input_ids,
            token_type_ids,
            attention_mask,
            visual_feats=visual_feats,
            visual_attention_mask=visual_attention_mask,
        )
        if "x" == self.mode:
            return pooled_output
        elif "x" in self.mode and ("l" in self.mode or "r" in self.mode):
            return feat_seq, pooled_output
        elif "l" in self.mode or "r" in self.mode:
            return feat_seq


class LXRTEncoder_(nn.Module):
    """
    Usage:
        Input:
            lxrt_encoder = LXRTEncoder(args,MAX_VQA_LENGTH=20).cuda()
            feat = torch.rand(128,36,2048).cuda()
            pos = torch.rand(128,36,4).cuda()
            sent = list(sentences) # len(sent) = batch_size i.e 128
        
        Output:
            output = lxrt_encoder(sent, (feat.cuda(), pos.cuda())) # [128,768]
    """

    def __init__(self, max_seq_length, params, mode="x"):
        super().__init__()
        self.max_seq_length = max_seq_length
        set_visual_config(params)

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )

        # Build LXRT Model
        self.model = VisualBertForLXRFeature.from_pretrained(
            "bert-base-uncased", params=params, mode=mode,
        )

        if params["from_scratch"]:
            print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)

    def multi_gpu(self):
        self.model = nn.DataParallel(self.model)

    @property
    def dim(self):
        return 768

    def forward(self, sents, feats, visual_attention_mask=None):

        train_features = convert_sents_to_features(
            sents, self.max_seq_length, self.tokenizer
        )

        input_ids = torch.tensor(
            [f.input_ids for f in train_features], dtype=torch.long
        ).to(
            device
        )  # [128,20]
        input_mask = torch.tensor(
            [f.input_mask for f in train_features], dtype=torch.long
        ).to(
            device
        )  # [128,20]
        segment_ids = torch.tensor(
            [f.segment_ids for f in train_features], dtype=torch.long
        ).to(
            device
        )  # [128,20]

        output = self.model(
            input_ids,
            segment_ids,
            input_mask,
            visual_feats=feats,
            visual_attention_mask=visual_attention_mask,
        )
        return output

    def save(self, path):
      torch.save(self.model.state_dict(), os.path.join("%s_LXRT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module.") :]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)


class VQAModel_Adaptive(nn.Module):
    def __init__(self, num_answers, params):
        super().__init__()

        # Build LXRT encoder
        self.lxrt_encoder = LXRTEncoder_(max_seq_length=MAX_VQA_LENGTH, params=params)
        hid_dim = self.lxrt_encoder.dim

        # VQA Answer heads
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            torch.nn.LayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers),
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        if params["adapt_span_enabled"]:
            print("Using Adaptive Variant")
        if params["sparse_enabled"]:
            print("Sparse Enabled")
        if params["layerdrop_enabled"]:
            print(
                "LayerDrop is enabled with dropping rate set to ",
                params["layerdrop_num_layers"],
            )

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size
        :param feat: (b, o, f) # [128, 36, 2048]
        :param pos:  (b, o, 4) # [128, 36, 4]
        :param sent: (b,) Type -- list of string # 128
        :param leng: (b,) Type -- int numpy array # [128, 3129]
        :return: (b, num_answer) The logit of each answers.
        """
        x = self.lxrt_encoder(sent, (feat.float(), pos.float()))
        logit = self.logit_fc(x)

        return logit
