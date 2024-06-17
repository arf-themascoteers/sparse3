from ds_manager import DSManager
from train_test_evaluator import evaluate_train_test_pair


def evaluate(dataset, folds, bands):
    oas = []
    aas = []
    ks = []
    d = DSManager(dataset,folds)
    for fold, splits in enumerate(d.get_k_folds()):
        evaluation_train_x = splits.evaluation_train_x[:,bands]
        evaluation_test_x = splits.evaluation_test_x[:,bands]

        oa, aa, k = evaluate_train_test_pair(evaluation_train_x, splits.evaluation_train_y, evaluation_test_x, splits.evaluation_test_y)
        oas.append(oa)
        aas.append(aa)
        ks.append(k)
    return oas, aas, ks


def compare(dataset, folds, bands1, bands2):
    oas1, aas1, ks1 = evaluate(dataset, folds, bands1)
    oas2, aas2, ks2 = evaluate(dataset, folds, bands2)

    mean_oas1 = sum(oas1)/len(oas1)
    mean_oas2 = sum(oas2)/len(oas2)

    if mean_oas1 > mean_oas2:
        print(f"First is better")
    else:
        print(f"Second is better")

    mean_aas1 = sum(aas1)/len(aas1)
    mean_aas2 = sum(aas2)/len(aas2)

    mean_k1 = sum(ks1)/len(ks1)
    mean_k2 = sum(ks2)/len(ks2)

    print(mean_oas1, mean_oas2)
    print(mean_aas1, mean_aas2)
    print(mean_k1, mean_k2)


dataset = "indian_pines"
folds = 10
bands1 = [10,16,24,27,40,47,53,59,63,78,83,92,100,106,119,120,129,140,145,153,154,166,176,182,190]
bands2 = [165,38,51,65,12,100,0,71,5,60,88,26,164,75,74,52,22,94,35,11,184,179,34,160,46]

compare(dataset, folds, bands1, bands2)



