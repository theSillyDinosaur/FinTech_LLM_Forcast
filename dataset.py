import json
import os
import torch
class Config:
    num_epochs = 10
    data_len = 4
    need_random = True

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)
week_set = [f"20{(37+i)//52+18}-week{(37+i)%52+1}" for i in range(0, 106)]
abbr_set = ["acc", "gfb", "glb", "gub", "shoe", "underwear"]
abbr_set_complete = ["accessories", "garment full body", "garment lower body", "garment upper body", "shoes", "underwear"]
text_dict = {}
for week in week_set:
    text_dict[week] = {}
    for i, abbr in enumerate(abbr_set):
        if os.path.exists(f"output/top5{abbr}/processed_{week}/top5{abbr}_{week}_descriptions.json"):
            with open(f"output/top5{abbr}/processed_{week}/top5{abbr}_{week}_descriptions.json", "r") as f:
                text_dict[week][abbr_set_complete[i]] = json.load(f)
        else:
            text_dict[week][abbr_set_complete[i]] = {}
        print(text_dict[week][abbr_set_complete[i]])
from torch.utils.data import Dataset
import random

random.seed(1126)

class TextDataset(Dataset):
    def __init__(self, text_dict, week_set):
        self.week_set = week_set
        self.week_dict = {week: i for i, week in enumerate(self.week_set)}
        self.abbr_set = list(text_dict[self.week_set[0]].keys())
        self.abbr_dict = {abbr: i for i, abbr in enumerate(self.abbr_set)}
        self.data = []
        for week in self.week_set[config.data_len:]:
            for abbr in self.abbr_set:
                valid_text = ""
                for prod in text_dict[week][abbr]:
                    for num in range(3):
                        valid_text += text_dict[week][abbr][prod][num] + " "
                    valid_text = valid_text[:-1] + "/"
                valid_text += "End of prediction."

                self.data.append({
                    "week": week,
                    "abbr": abbr,
                    "valid_text": valid_text,
                  })
        self.text_dict = text_dict
    def __len__(self):
        return len(self.data)*config.num_epochs


    def __getitem__(self, idx):
        query = "Below are the descriptions for the previous top 5 best-sellers of each types."
        item = self.data[idx//config.num_epochs]
        prev_len = random.randint(config.data_len//2, config.data_len+1)
        for i in range(self.week_dict[item["week"]]-prev_len, self.week_dict[item["week"]]):
            abbr = item["abbr"]
            query += f"Best sellers in {self.week_set[i]}: "
            prod_set = list(self.text_dict[self.week_set[i]][abbr].keys())
            if config.need_random:
                random.shuffle(prod_set)
            for j, prod in enumerate(prod_set):
                query += f"{self.text_dict[self.week_set[i]][abbr][prod][0]}, {self.text_dict[self.week_set[i]][abbr][prod][1]}, {self.text_dict[self.week_set[i]][abbr][prod][2]}/"
            query += "End of prediction."
        query += f"You are a best seller predictor, and above are the descriptions for the previous top 5 best-sellers of each types. Do your best to predict the description of the best sellers of {item['abbr']} in {item['week']} based on the previous information. One minimun and five maximum. Split the description with '/' and end the response in 'End of prediction.' "
        response = f"{item['valid_text']}"


        output = f"""USER: {query}\nASSISTANT: {response}"""



        return output


# Instantiate the dataset
train_dataset = TextDataset(text_dict, week_set[:75])
test_dataset = TextDataset(text_dict, week_set[75:])

seq_beforeHug = {
    "train": [],
    "test": []
}
for i in range(len(train_dataset)):
    seq_beforeHug["train"].append({"text": train_dataset[i]})
for i in range(len(test_dataset)):
    seq_beforeHug["test"].append({"text":test_dataset[i]})
