def zero_if_exception(scorer):
    def new_scorer(*args, **kwargs):
        try:
            return scorer(*args, **kwargs)
        except:
            return 0

    return new_scorer

@zero_if_exception
def get_score(y_true, prediction):
    if "correct" in y_true:
        if y_true["correct"] == prediction:
            return 1
    elif "correct_variants" in y_true and isinstance(
        y_true["correct_variants"][0], str
    ):
        if prediction in y_true["correct_variants"]:
            return 1
    elif "correct_variants" in y_true and isinstance(
        y_true["correct_variants"][0], list
    ):
        y_true = set(y_true["correct_variants"][0])
        y_pred = set(prediction)
        return int(
            len(set.intersection(y_true, y_pred)) == len(y_true) == len(y_pred)
        )
    return 0