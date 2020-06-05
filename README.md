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

1. Запустить эксперименты на бОльшем кол-ве эпох.

1. Улучшить результат на задаче 18.<br>
Можно попробовать другой подход при обучении или вариант, предложенный куратором.<br>
Было предложение попробовать следующее ...

```
По 17-20 могу сразу сказать, что я бы это решал как разметку последовательности.
То есть надо взять конфиг для морфологии из DeepPavlov, только в качестве метки предсказывать не то, что в тех задачах, а знаки препинания.
Ссылка на документацию: http://docs.deeppavlov.ai/en/master/features/models/bert.html#bert-for-named-entity-recognition-sequence-tagging,
конфиг https://github.com/deepmipt/DeepPavlov/blob/master/deeppavlov/configs/morpho_tagger/BERT/morpho_ru_syntagrus_bert.json но там нужно заменить reader.
Поскольку deeppavlov это скорее tf, можно взять реализацию разметки сущностей на торче и ориентироваться на неё https://github.com/huggingface/transformers/tree/master/examples/ner,
формально задача та же — разметка последовательности.
```


## Main_Text_Idea

Main_Text_Idea решает задачу №1 ЕГЭ.

Цель: указать номера предложений, в которых передана главная информация, содержащаяся в тексте
Тип задания: multiple_choice
Постановка задачи:
- Собрать дополнительные данные с различных источников: Yandex, fipi, etc.
- Найти предложения, используя косинусную близость между эмбеддингами вариантов ответа на уровне предложения, полученных с помощью BertEmbedder

Участники: `MTETERIN`




Было собрано приблизительно 200 дополнительных вопросов по задаче поиска главной информации в тексте.

Было принято решение попробовать несколько BERT моделей:
- DeepPavlov/rubert-base-cased
- DeepPavlov/rubert-base-cased-sentence
- DeepPavlov/bert-base-multilingual-cased-sentence
- bert-base-multilingual-cased
- xlm-roberta-base
- xlm-robert-large

### Эксперименты
### Результаты экспериментов

| model | layer number | model accuracy |
|-------|------------------------|---------------------|
|winner-solution (bert-base-multilingual-cased) | 12 (MEAN) | 86% |
| bert-base-multilingual-cased | isnextSentencePrediction | 6.55% |
| bert-base-multilingual-cased | 12 (CLS) | 45% |
| bert-base-multilingual-cased | 11 (CLS) | 46,2% |
| bert-base-multilingual-cased | 10 (CLS) | 46,2% |
| bert-base-multilingual-cased | 9 (CLS) | 54,3% |
| bert-base-multilingual-cased | 8 (CLS) | 53,7% |
| bert-base-multilingual-cased | 7 (CLS) | 52,6% |
| bert-base-multilingual-cased | 6 (CLS) | 58,9% |
| bert-base-multilingual-cased | 5 (CLS) | 64,5% |
| bert-base-multilingual-cased | 4 (CLS) | 77,9% |
| bert-base-multilingual-cased | 3 (CLS) | 74,7% |
| bert-base-multilingual-cased | 2 (CLS) | 65,1% |
| bert-base-multilingual-cased | 1 (CLS) | 57,8% |
| bert-base-multilingual-cased | 0 (CLS) | 0,005% |
| DeepPavlov/rubert-base-cased | 4 (CLS) | 74,7% |
| DeepPavlov/rubert-base-cased-sentence | 5 (CLS) | 72,5% |
| DeepPavlov/rubert-base-cased-sentence | 4 (CLS) | 80,1% |
| DeepPavlov/rubert-base-cased-sentence | 3 (CLS) | 74,7% |
| bert-base-multilingual-cased | 11 (MEAN) | 87% |
| bert-base-multilingual-cased | 10 (MEAN) | 88% |
| bert-base-multilingual-cased | 9 (MEAN) | 89% |
| bert-base-multilingual-cased | 8 (MEAN) | 99% |
| bert-base-multilingual-cased | 7 (MEAN) | 89% |
| bert-base-multilingual-cased | 6 (MEAN) | 89,2% |
| bert-base-multilingual-cased (BEST) | 5 (MEAN) | 90,3% |
| bert-base-multilingual-cased | 4 (MEAN) | 89,2% |
| bert-base-multilingual-cased | 3 (MEAN) | 88,7% |
| bert-base-multilingual-cased | 2 (MEAN) | 88,2% |
| bert-base-multilingual-cased | 1 (MEAN) | 86% |
| bert-base-multilingual-cased | 0 (MEAN) | 86% |
| bert-base-multilingual-cased FULL-TEXT + OPTIONS | 12 (MEAN) | 50% |
| bert-base-multilingual-cased FULL-TEXT + OPTIONS | 6 (MEAN) | 54% |
| bert-base-multilingual-cased FULL-TEXT + OPTIONS | 5 (MEAN) | 55,2% |
| bert-base-multilingual-cased FULL-TEXT + OPTIONS | 0 (MEAN) | 51,2% |


### Выводы

Все модели показали приблизительно одинаковую точность. К сожалению не удалось получить хорошую точность используя информацию о тексте из которого нужно извлечь главную информацию.
Самые лучшие результаты показали 55%. Самая лучшая модель с учетом вариантов ответа была bert-base-multilingual-cased  на 5 слоем - 90.3%.
 
В дальнейшем хотелось бы повысить точность модели (FULL-TEXT + OPTIONS). Скорей всего этот текст нужно как-то уменьшать. Возможно через суммаризацию.


Литература:
- https://arxiv.org/abs/1810.04805
- https://medium.com/analytics-vidhya/semantic-similarity-in-sentences-and-bert-e8d34f5a4677
- https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
- https://medium.com/@adriensieg/text-similarities-da019229c894




## Word sense disambiguation

Word sense disambiguation rue решает задачу 3 из ЕГЭ.

Описание задания: Прочитайте фрагмент словарной статьи, в которой приводятся значения слова WORD. Определите значение, в котором это слово употреблено во втором (k) предложении текста. Выпишите цифру, соответствующую этому значению в приведённом фрагменте словарной статьи.

Участники: `zhav1k`

Эта задача на сравнение схожести эмбеддингов.

Литература:
- https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
- https://www.markiiisys.com/blog/finding-cosine-similarity-between-sentences-using-bert-as-a-service/
- https://medium.com/analytics-vidhya/semantic-similarity-in-sentences-and-bert-e8d34f5a4677
- https://towardsdatascience.com/nlp-extract-contextualized-word-embeddings-from-bert-keras-tf-67ef29f60a7b
- http://jalammar.github.io/illustrated-bert/

### Эксперименты

Модели:
- DeepPavlov/rubert-base-cased
- bert-base-multilingual-cased

Слои:
- 11
- с 8 по 11
- с 0 по 11
- 9 и 11

Как представлять вектор:
- конкатенация <CLS> токенов
- среднее <CLS> токенов
- конкатенация всех токенов
- среднее всех токенов

Косинусная мера между:
- эмбеддингом текста и эмбеддингами вариантов ответа
- эмбеддингом ключевого предложения и эмбеддингами вариантов ответа
- контекстуальным эмбеддингом слова в предложении и эмбеддингами вариантов ответа

Метрика - Accuracy на тесте.

Так как различных комбинаций аж 72 штуки, приведены основные результаты.

Также я пробовал чистить дополнительно текст от разных вставок и символов (цифр предложений, пропущенный текст в виде <...>). Первое никакого результата не дает,
а вот в эмбеддингах предложений <...> играет некую роль, т.к результат становится хуже. (1)

Также пробовал посмотреть взаимосвязь между "предсказаниями" разных варинатов эмбеддингов, с целью выяснить, есть ли смысл пытаться
взвешивать результаты косинусной близости между методами (своего рода ансамбль). Однако результаты косинусной близости оказывались сильно  
коррелированными, и я не увидел смысла тюнить веса на методы.

### Основные результаты

| model | layers | represent layers | cosine between | accuracy test |
|-------------------|---------|------------|--------|-----|
| bert-bert-base-multilingual-cased | 8 - 11 | concat CLS | text variants | 36% |
| bert-bert-base-multilingual-cased | 9 11 | concat all | sentence variants | 41% |
| bert-bert-base-multilingual-cased | 11 | no | contextualized word embedding variants | 46% |
| bert-bert-base-multilingual-cased | 9 11 | mean | contextualized word embedding variants | 48.57% |
| bert-bert-base-multilingual-cased | 0 - 11 | mean | contextualized word embedding variants | 48.57% |
| DeepPavlov/rubert-base-cased | 9 11 | mean | contextualized word embedding variants | 40% |
| DeepPavlov/rubert-base-cased | 0 - 11 | mean | contextualized word embedding variants | 54.28% |

### Выводы

Выводы: задача оказалась достаточно сложной, по моему мнению в ней присутствует большая доля стохастики, так как от изменения пары символов, могло меняться качество предсказаний. см. (1)
Поэтому один из выводов, и, возможно, вариант дальнейшего действия - чистка/обработка данных (хотя пока плохо представляю, что в этом направлении еще можно сделать).

Также интересно заметить, что для RuBert и multilingual BERT были оптимальны разные комбинации hidden layers. + всегда стоит пробовать разновидности BERT (RuBert в моем случае), даже если говорят, что он обычно не выстреливает:)
Лучшими эмбеддингами под задачу оказись эмбеддинги из DeepPavlov/rubert-base-cased


### Что дальше?

1. Пробовать чистить данные еще больше.

2. Пробовать более большие модели (ex. xlm-roberta)


