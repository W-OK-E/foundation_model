met = {"class_0_iou": 0.0, "class_0_dice": 0.0, "class_0_f": 0.0, "class_0_precision": 0.0, "class_0_recall": 0.0, "class_1_iou": 66.30962652952974, "class_1_dice": 79.74237921550005, "class_1_f": 79.74237921550005, "class_1_precision": 100.0, "class_1_recall": 66.30962652952974}
mean = []
for k,v in met.items():
    if("precision" in k):
        mean.append(v)

print(sum(mean)/(len(mean)-1))

