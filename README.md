# final-exams-auto-solver

## Punctuator

Punctuator решает задачки 17-20 ЕГЭ.

Описание задания: указать цифры, на месте которых в предложении должны стоять запятые (multiple_choice)

Постановка задачи:
- `task 17` Расставьте знаки препинания: укажите цифру(-ы), на месте которой(-ых) в предложении должна(-ы) стоять запятая(-ые)
- `task 18` Расставьте все недостающие знаки препинания: укажите цифру(-ы), на месте которой(-ых) в предложениях должна(-ы) стоять запятая(-ые)
- `task 19` Расставьте все знаки препинания: укажите цифру(-ы), на месте которой(-ых) в предложении должна(-ы) стоять запятая(-ые)
- `task 20` Расставьте знаки препинания: укажите цифру(-ы), на месте которой(-ых) в предложении должна(-ы) стоять запятая(-ые)

Участники: `LeonidMorozov`

По моему первоначальному мнению, эта задача для Bert Masked LM.

Литература:
- https://arxiv.org/abs/1810.04805
- https://medium.com/saarthi-ai/bert-how-to-build-state-of-the-art-language-models-59dddfa9ac5d

Единственная проблема -- маленький изначальный датасет (выданный с самим соревнованием),
но в моей задаче это решается достаточно просто.
Подойдут любые валидные тексты (например из худ. литературы),
т.е. любые достаточно длинные предложения, содержащие хотя бы одну запятую.

Простой скрипт из punctuator/extra_dataset/punctuator_dataset_parser.ipynb дал 6-7K новых dataset samples


### Эксперименты

Было принято решение попробовать несколько BERT моделей:
- DeepPavlov/rubert-base-cased
- DeepPavlov/rubert-base-cased-sentence
- DeepPavlov/bert-base-multilingual-cased-sentence
- bert-base-multilingual-cased
- bert-base-multilingual-uncased

А так же 2 варианта обучения:
- одна модель на все 4 задачи
- отдельная модель на каждую из задач 17-20

Используем Accuracy на Val как основную метрику


### Результаты одной модели на все 4 задачи

| model | best model train epoch | best model accuracy |
|-------|------------------------|---------------------|
| DeepPavlov/rubert-base-cased | 5 | 98.55% |
| DeepPavlov/rubert-base-cased-sentence | 8 | 98.67% |
| DeepPavlov/bert-base-multilingual-cased-sentence | 8 | 98.24% |
| bert-base-multilingual-cased | 10 | 98.63% |
| bert-base-multilingual-uncased | 9 | 98.55% |

### Результаты отдельной модели на каждой из 4х задач

| task | model | best model train epoch | best model accuracy |
|------|-------|------------------------|---------------------|
| 17 | DeepPavlov/rubert-base-cased | 8 | 98.72% |
| 17 | DeepPavlov/rubert-base-cased-sentence | 5 | 98.88% |
| 17 | DeepPavlov/bert-base-multilingual-cased-sentence | 8 | 98.08% |
| 17 | bert-base-multilingual-cased | 7 | 98.80% |
| 17 | bert-base-multilingual-uncased | 3 | 98.56% |
| 18 | DeepPavlov/rubert-base-cased | 6 | 95.65% |
| 18 | DeepPavlov/rubert-base-cased-sentence | 5 | 95.45% |
| 18 | DeepPavlov/bert-base-multilingual-cased-sentence | 7 | 92.87% |
| 18 | bert-base-multilingual-cased | 10 | 94.09% |
| 18 | bert-base-multilingual-uncased | 6 | 94.70% |
| 19 | DeepPavlov/rubert-base-cased | 5 | 99.39% |
| 19 | DeepPavlov/rubert-base-cased-sentence | 5 | 99.31% |
| 19 | DeepPavlov/bert-base-multilingual-cased-sentence | 8 | 99.65% |
| 19 | bert-base-multilingual-cased | 3 | 99.22% |
| 19 | bert-base-multilingual-uncased | 9 | 99.65% |
| 20 | DeepPavlov/rubert-base-cased | 4 | 98.91% |
| 20 | DeepPavlov/rubert-base-cased-sentence | 8 | 98.52% |
| 20 | DeepPavlov/bert-base-multilingual-cased-sentence | 6 | 98.44% |
| 20 | bert-base-multilingual-cased | 4 | 98.67% |
| 20 | bert-base-multilingual-uncased | 4 | 98.44% |


### Выводы

Лучшая модель на всех 4х задачах -- DeepPavlov/rubert-base-cased и DeepPavlov/rubert-base-cased-sentence

Лучшая модель на задачах по-отдельности -- DeepPavlov/rubert-base-cased

Видно, что в некоторых случаях 10 ep не достаточно для обучения, надо повторить эксперименты на 20-30 ep.

Интересно, что bert-base-multilingual-uncased выдал очень похожий результат на DeepPavlov/rubert-base-cased


На самом деле все модели показали примерно похожие результаты, каким-то моделям нужно было больше epoch для достижения max accuracy.

Что характерно, все модели испытывали трудности с задачей 18.


### Что дальше?

Улучшить результат на задаче 18.

Можно попробовать другой подход при обучении или вариант, предложенный куратором.

Было предложение попробовать следующее ...

```
По 17-20 могу сразу сказать, что я бы это решал как разметку последовательности.
То есть надо взять конфиг для морфологии из DeepPavlov, только в качестве метки предсказывать не то, что в тех задачах, а знаки препинания.
Ссылка на документацию: http://docs.deeppavlov.ai/en/master/features/models/bert.html#bert-for-named-entity-recognition-sequence-tagging,
конфиг https://github.com/deepmipt/DeepPavlov/blob/master/deeppavlov/configs/morpho_tagger/BERT/morpho_ru_syntagrus_bert.json но там нужно заменить reader.
Поскольку deeppavlov это скорее tf, можно взять реализацию разметки сущностей на торче и ориентироваться на неё https://github.com/huggingface/transformers/tree/master/examples/ner,
формально задача та же — разметка последовательности.
```
