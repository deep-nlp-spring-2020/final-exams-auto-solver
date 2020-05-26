import time
from model import MainTextIdea
from data import read_config
from utils import get_score





if __name__ == '__main__':
    DATA_PATH = './data'
    data = read_config(f'{DATA_PATH}/test.json')
    solver = MainTextIdea()
    scores = 0
    max_scores = len(data)
    for i, task in enumerate(data):
        start = time.time()
        task_index, task_type = i + 1, 'multiple_choice'
        print("Predicting task {}...".format(task_index))
        y_true = task["solution"]
        try:
            prediction = solver.predict_from_model(task)
        except BaseException as e:
            print(e)
            print("Solver {} failed to solve task №{}".format('1', task_index))
            prediction = ""
        score = get_score(y_true, prediction)
        scores += score
        print(
            "Score: {}\nCorrect: {}\nPrediction: {}\n".format(
                score, y_true, prediction
            )
        )
    print(f'max_scores={max_scores}, scores={scores}')