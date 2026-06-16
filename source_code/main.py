# # %%
# import torch
# import string

# from transformers import BertTokenizer, BertForMaskedLM
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()

# from transformers import XLNetTokenizer, XLNetLMHeadModel
# xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
# xlnet_model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased').eval()

# from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
# xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
# xlmroberta_model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base').eval()

# from transformers import BartTokenizer, BartForConditionalGeneration
# bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
# bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').eval()

# from transformers import ElectraTokenizer, ElectraForMaskedLM
# electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
# electra_model = ElectraForMaskedLM.from_pretrained('google/electra-small-generator').eval()

# from transformers import RobertaTokenizer, RobertaForMaskedLM
# roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base').eval()

# top_k = 10


# def decode(tokenizer, pred_idx, top_clean):
#     ignore_tokens = string.punctuation + '[PAD]'
#     tokens = []
#     for w in pred_idx:
#         token = ''.join(tokenizer.decode(w).split())
#         if token not in ignore_tokens:
#             tokens.append(token.replace('##', ''))
#     return '\n'.join(tokens[:top_clean])


# # def encode(tokenizer, text_sentence, add_special_tokens=True):
# #     text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
# #     # if <mask> is the last token, append a "." so that models dont predict punctuation.
# #     if tokenizer.mask_token == text_sentence.split()[-1]:
# #         text_sentence += ' .'

# #     input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
# #     mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
# #     return input_ids, mask_idx

# def encode(tokenizer, text_sentence, add_special_tokens=True):
#     text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    
#     # Check if mask token exists
#     if tokenizer.mask_token not in text_sentence:
#         raise ValueError("Mask token not found in the input sentence.")
    
#     # If <mask> is the last token, append a "." so that models don't predict punctuation.
#     if tokenizer.mask_token == text_sentence.split()[-1]:
#         text_sentence += ' .'

#     input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    
#     # Get the index of the mask token in the input
#     mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    
#     return input_ids, mask_idx



# def get_all_predictions(text_sentence, top_clean=5):
#     # ========================= BERT =================================
#     print(text_sentence)
#     input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
#     with torch.no_grad():
#         predict = bert_model(input_ids)[0]
#     bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

#     # ========================= XLNET LARGE =================================
#     input_ids, mask_idx = encode(xlnet_tokenizer, text_sentence, False)
#     perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
#     perm_mask[:, :, mask_idx] = 1.0  # Previous tokens don't see last token
#     target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
#     target_mapping[0, 0, mask_idx] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

#     with torch.no_grad():
#         predict = xlnet_model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)[0]
#     xlnet = decode(xlnet_tokenizer, predict[0, 0, :].topk(top_k).indices.tolist(), top_clean)

#     # ========================= XLM ROBERTA BASE =================================
#     input_ids, mask_idx = encode(xlmroberta_tokenizer, text_sentence, add_special_tokens=True)
#     with torch.no_grad():
#         predict = xlmroberta_model(input_ids)[0]
#     xlm = decode(xlmroberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

#     # ========================= BART =================================
#     input_ids, mask_idx = encode(bart_tokenizer, text_sentence, add_special_tokens=True)
#     with torch.no_grad():
#         predict = bart_model(input_ids)[0]
#     bart = decode(bart_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

#     # ========================= ELECTRA =================================
#     input_ids, mask_idx = encode(electra_tokenizer, text_sentence, add_special_tokens=True)
#     with torch.no_grad():
#         predict = electra_model(input_ids)[0]
#     electra = decode(electra_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

#     # ========================= ROBERTA =================================
#     input_ids, mask_idx = encode(roberta_tokenizer, text_sentence, add_special_tokens=True)
#     with torch.no_grad():
#         predict = roberta_model(input_ids)[0]
#     roberta = decode(roberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

#     return {'bert': bert,
#             'xlnet': xlnet,
#             'xlm': xlm,
#             'bart': bart,
#             'electra': electra,
#             'roberta': roberta}


import torch
import string
from transformers import BertTokenizer, BertForMaskedLM
from transformers import XLNetTokenizer, XLNetLMHeadModel
from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import ElectraTokenizer, ElectraForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM

# Load pre-trained models and tokenizers
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()

xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
xlnet_model = XLNetLMHeadModel.from_pretrained('xlnet-base-cased').eval()

xlmroberta_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
xlmroberta_model = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base').eval()

bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large').eval()

electra_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-generator')
electra_model = ElectraForMaskedLM.from_pretrained('google/electra-small-generator').eval()

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base').eval()

# Maximum number of predictions to return
top_k = 10

def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])

def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # If <mask> is the last token, append a "." so that models don't predict punctuation
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]

    # Debugging: Print input_ids and mask_idx
    print(f"Encoded input_ids: {input_ids}")
    print(f"Mask index: {mask_idx}")

    return input_ids, mask_idx

def get_all_predictions(text_sentence, top_clean=5):
    print(f"Predictions requested for: {text_sentence}")
    
    try:
        # ========================= BERT =================================
        input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
        with torch.no_grad():
            predict = bert_model(input_ids)[0]
        print(f"BERT prediction shape: {predict.shape}")

        if predict.shape[1] <= mask_idx:
            print("Warning: Mask index is out of range for BERT.")
            return {'error': 'Mask index out of range for BERT'}

        bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
        print(f"BERT predictions: {bert}")

        # ========================= XLNET =================================
        input_ids, mask_idx = encode(xlnet_tokenizer, text_sentence, False)
        perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
        perm_mask[:, :, mask_idx] = 1.0
        target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)
        target_mapping[0, 0, mask_idx] = 1.0

        with torch.no_grad():
            predict = xlnet_model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)[0]
        xlnet = decode(xlnet_tokenizer, predict[0, 0, :].topk(top_k).indices.tolist(), top_clean)
        print(f"XLNet predictions: {xlnet}")

        # ========================= Other Models (similar to above) ==========================
        # XLM, BART, ELECTRA, ROBERTA...

        return {
            'bert': bert,
            'xlnet': xlnet,
            # Add other models like 'xlm', 'bart', 'electra', etc.
        }

    except Exception as e:
        print(f"Error in prediction: {e}")
        return {'error': str(e)}

