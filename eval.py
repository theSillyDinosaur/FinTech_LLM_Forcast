from dataset import test_dataset, device
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
import open_clip
import numpy as np
import matplotlib.pyplot as plt


test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/vicuna-bestseller-lora-1abbr/checkpoint-1065", quantization_config=bnb_config)
model.eval() # Set the model to evaluation mode

# Load CLIP model and processor
# Using the Hugging Face model directly
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Or using open_clip (often preferred for flexibility)
# model_clip, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')

model_clip.eval()
model_clip.to(device)

def calculate_clip_score(actuals, predictions, model, processor, device):
    clip_scores = []
    clip_best_scores = []
    with torch.no_grad():
        for actual_set, pred_set in zip(actuals, predictions):
            actual_set = actual_set.split('End of prediction')[0].split('/')
            pred_set = pred_set.split('End of prediction')[0].split('/')
            for pred in pred_set:
                best_score = 0
                for actual in actual_set:
                    # Process text inputs
                    inputs = processor(text=[actual, pred], return_tensors="pt", padding=True, truncation=True).to(device)

                    # Get text embeddings
                    text_features = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

                    # Normalize embeddings
                    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)

                    # Calculate cosine similarity (CLIP score)
                    # The score is the dot product of the normalized embeddings
                    score = torch.dot(text_features_norm[0], text_features_norm[1]).item()
                    best_score = max(best_score, score)
                    clip_scores.append(score)
                clip_best_scores.append(best_score)

    return clip_scores, clip_best_scores

predictions = []
actuals = []
model.to(device)

with torch.no_grad():
    clip_scores = []
    clip_best_scores = []
    for i in range(0, len(test_dataset), 13):
        data = [test_dataset[i]]
        input_data = [sentence.split("ASSISTANT:")[0]+"\nASSISTANT:" for sentence in data]
        inputs = tokenizer(input_data, return_tensors="pt", padding=True, truncation=True, max_length=25000).to(device)
        outputs = model.generate(**inputs, max_new_tokens=100, num_return_sequences=1)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the response part from the decoded output
        # This assumes your output format is "USER: query\nASSISTANT: response"
        response_start_index = decoded_output.find("ASSISTANT:")
        if response_start_index != -1:
            predicted_response = decoded_output[response_start_index + len("ASSISTANT:"):].strip()
            predictions.append(predicted_response)
        else:
            predictions.append("") # Handle cases where the format is unexpected

        # Extract the actual response from the original test data
        # This assumes the test data format is "USER: query\nASSISTANT: response"
        actual_response_start_index = data[0].find("ASSISTANT:")
        if actual_response_start_index != -1:
            actual_response = data[0][actual_response_start_index + len("ASSISTANT:"):].strip()
            actuals.append(actual_response)
        else:
            actuals.append("") # Handle cases where the format is unexpected
        clip_score, clip_best_score = calculate_clip_score([actual_response], [predicted_response], model_clip, processor, device)
        print(f"Sample {i//13+1}:")
        print(f"  Predicted: {predicted_response}")
        print(f"  Actual:    {actual_response}")
        print(f"  Clip_Score: {clip_score, clip_best_score}")
        print("-" * 20)
        clip_scores += clip_score
        clip_best_scores += clip_best_score


# Ensure clip_best_scores and clip_scores are numpy arrays for easier manipulation
clip_best_scores = np.array(clip_best_scores)
clip_scores = np.array(clip_scores)
print(clip_best_scores)
print(clip_scores)

# Calculate CDF for clip_best_scores
# Sort data in descending order for inverse plot
sorted_best_scores = np.sort(clip_best_scores)[::-1]
sorted_scores = np.sort(clip_scores)[::-1]
cdf_best = np.arange(1, len(sorted_best_scores) + 1) / len(sorted_best_scores)
cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

# Plotting the inverse CDF
plt.figure(figsize=(10, 6))
plt.plot(sorted_best_scores, cdf_best, marker='.', linestyle='none')
plt.xlabel('CLIP Best Score')
plt.ylabel('CDF')
plt.title('Cumulative Distribution Function of CLIP Best Scores (Inverse X-axis)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sorted_scores, cdf, marker='.', linestyle='none')
plt.xlabel('CLIP Score')
plt.ylabel('CDF')
plt.title('Cumulative Distribution Function of CLIP Best Scores (Inverse X-axis)')
plt.grid(True)
plt.show()

# Calculate the portion of scores > 0.7, > 0.8, and > 0.9 for clip_best_scores
portion_gt_07_best = np.sum(clip_best_scores > 0.7) / len(clip_best_scores)
portion_gt_08_best = np.sum(clip_best_scores > 0.8) / len(clip_best_scores)
portion_gt_09_best = np.sum(clip_best_scores > 0.9) / len(clip_best_scores)
portion_gt_07 = np.sum(clip_scores > 0.7) / len(clip_scores)
portion_gt_08 = np.sum(clip_scores > 0.8) / len(clip_scores)
portion_gt_09 = np.sum(clip_scores > 0.9) / len(clip_scores)

print(f"Portion of CLIP Best Scores > 0.7: {portion_gt_07_best:.4f}")
print(f"Portion of CLIP Best Scores > 0.8: {portion_gt_08_best:.4f}")
print(f"Portion of CLIP Best Scores > 0.9: {portion_gt_09_best:.4f}")
print(f"Portion of CLIP Scores > 0.7: {portion_gt_07:.4f}")
print(f"Portion of CLIP Scores > 0.8: {portion_gt_08:.4f}")
print(f"Portion of CLIP Scores > 0.9: {portion_gt_09:.4f}")