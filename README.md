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
| DeepPavlov/rubert-base-cased | 19 | 98.88% |
| DeepPavlov/rubert-base-cased-sentence | 19 | 98.80% |
| DeepPavlov/bert-base-multilingual-cased-sentence | 19 | 98.19% |
| bert-base-multilingual-cased | 16 | 98.51% |
| bert-base-multilingual-uncased | 18 | 98.61% |

### Результаты отдельной модели на каждой из 4х задач

| task | model | best model train epoch | best model accuracy |
|------|-------|------------------------|---------------------|
| 17 | DeepPavlov/rubert-base-cased | 5 | 98.88% |
| 17 | DeepPavlov/rubert-base-cased-sentence | 9 | 98.48% |
| 17 | DeepPavlov/bert-base-multilingual-cased-sentence | 21 | 98.00% |
| 17 | bert-base-multilingual-cased | 5 | 98.40% |
| 17 | bert-base-multilingual-uncased | 12 | 98.80% |
| 18 | DeepPavlov/rubert-base-cased | 6 | 96.60% |
| 18 | DeepPavlov/rubert-base-cased-sentence | 16 | 93.21% |
| 18 | DeepPavlov/bert-base-multilingual-cased-sentence | 14 | 93.21% |
| 18 | bert-base-multilingual-cased | 12 | 95.04% |
| 18 | bert-base-multilingual-uncased | 5 | 95.92% |
| 19 | DeepPavlov/rubert-base-cased | 10 | 99.39% |
| 19 | DeepPavlov/rubert-base-cased-sentence | 13 | 99.13% |
| 19 | DeepPavlov/bert-base-multilingual-cased-sentence | 8 | 99.22% |
| 19 | bert-base-multilingual-cased | 20 | 99.39% |
| 19 | bert-base-multilingual-uncased | 4 | 99.22% |
| 20 | DeepPavlov/rubert-base-cased | 14 | 99.22% |
| 20 | DeepPavlov/rubert-base-cased-sentence | 14 | 98.12% |
| 20 | DeepPavlov/bert-base-multilingual-cased-sentence | 8 | 98.59% |
| 20 | bert-base-multilingual-cased | 17 | 98.98% |
| 20 | bert-base-multilingual-uncased | 17 | 98.83% |


### Выводы

Лучшая модель на всех 4х задачах -- DeepPavlov/rubert-base-cased и DeepPavlov/rubert-base-cased-sentence

Лучшая модель на задачах по-отдельности -- DeepPavlov/rubert-base-cased

Видно, что в некоторых случаях 10 ep не достаточно для обучения, надо повторить эксперименты на 20-30 ep.

Интересно, что bert-base-multilingual-uncased выдал очень похожий результат на DeepPavlov/rubert-base-cased


На самом деле все модели показали примерно похожие результаты, каким-то моделям нужно было больше epoch для достижения max accuracy.

Что характерно, все модели испытывали трудности с задачей 18.


### Что дальше?

1. Запустить эксперименты на бОльшем кол-ве эпох (сделано, ноутбук с экспериментами и отчет обновлены).

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




## word_sense_disambiguation

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

Также я пробовал чистить дополнительно текст от разных вставок и символов (цифр предложений, пропущенный текст в виде <...>). Первое никакого результата не дает,
а вот в эмбеддингах предложений <...> играет некую роль, т.к результат становится хуже. (1)

Пробовал посмотреть взаимосвязь между "предсказаниями" разных варинатов эмбеддингов, с целью выяснить, есть ли смысл пытаться
взвешивать результаты косинусной близости между методами (своего рода ансамбль). Однако результаты косинусной близости оказывались сильно  
коррелированными, и я не увидел смысла тюнить веса на методы.

Так как различных комбинаций аж 72 штуки, приведены основные результаты.
### Основные результаты

| model | layers | represent layers | cosine between | accuracy test |
|------------------------|---------|------------|--------|-----|
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


# Kaggle соревнование jigsaw-multilingual-toxic-comment-classification

https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification

Описание задания:

It only takes one toxic comment to sour an online discussion. The Conversation AI team, a research initiative founded by Jigsaw and Google, builds technology to protect voices in conversation. A main area of focus is machine learning models that can identify toxicity in online conversations, where toxicity is defined as anything rude, disrespectful or otherwise likely to make someone leave a discussion. If these toxic contributions can be identified, we could have a safer, more collaborative internet.

In the previous 2018 Toxic Comment Classification Challenge, Kagglers built multi-headed models to recognize toxicity and several subtypes of toxicity. In 2019, in the Unintended Bias in Toxicity Classification Challenge, you worked to build toxicity models that operate fairly across a diverse range of conversations. This year, we're taking advantage of Kaggle's new TPU support and challenging you to build multilingual models with English-only training data.

Jigsaw's API, Perspective, serves toxicity models and others in a growing set of languages (see our documentation for the full list). Over the past year, the field has seen impressive multilingual capabilities from the latest model innovations, including few- and zero-shot learning. We're excited to learn whether these results "translate" (pun intended!) to toxicity classification. Your training data will be the English data provided for our previous two competitions and your test data will be Wikipedia talk page comments in several different languages.

As our computing resources and modeling capabilities grow, so does our potential to support healthy conversations across the globe. Develop strategies to build effective multilingual models and you'll help Conversation AI and the entire industry realize that potential.


Участники: `LeonidMorozov`, `MTETERIN`

Литература:
- https://arxiv.org/abs/1810.04805
- https://arxiv.org/abs/1907.11692
- https://en.wikipedia.org/wiki/Tf-idf
- https://en.wikipedia.org/wiki/Logistic_regression


# Dataset

Датасет -- мы попробовали взять свободный датасет RuTweetCorp (http://study.mokoron.com/) и разметить его.
Датасет состоит из 2 частей: positive и negative.
Мы решили попробовать разметить часть датасета и посмотреть результаты. 
Для этого мы взяли по 1000 самых длинных сообщений из positive и negative.
Так же мы стали размечать по toxic levels -- 1,2,3,4,5
Это дало нам некоторую свободу в том, какой level наиболее близкий к toxic из оригинальных датасетов. 
В итоге наиболее близкий результат дали tocix = level > 2 
Но общие результаты с этим датасетом оказались не очень, и мы решили просто перевести оригинальный en-датасет на нужные языки.
Для этого мы воспользовалить MarianMTModel и перевели EN toxic комментарии на ES, PT, TR, IT, FR и даже RU


### Эксперименты

Мы взяли вот этот kernel как baseline: https://www.kaggle.com/shonenkov/tpu-training-super-fast-xlmrobe

Мы так же провели множество экспериментов с data pre-processing, обучения BERT models (остановились на bert-base-multilingual-cased)
Так же TF-IDF с LogReg дали достаточно хорошие результаты.

| experiment | score | bump |
|------------|-------|------|
| Baseline RoBERTa Kernel | 94.16 | +0.0 |
| RoBERTa + BERT | 94.32 | +0.16 | 
| RoBERTa + BERT + TF-IDF | 94.39 | +0.07 |
| RoBERTa + BERT + TF-IDF + extra-datasets | 94.50 | +0.11 |
| fine-tune TF-IDF | 94.66 | +0.06 |
| fine-tune Ensemble | 94.70 | +0.04 |

Data pre-processing, который показали наилучший результат
- data shuffling
- remove duplication records with probability 0.95
- remove numbers from texts with probability 0.95
- remove hashtags from texts with probability 0.95 • remove urls from texts with probability 0.95
- remove usernames from texts with probability 0.95

### Выводы

1. оригинальный датасет достаточно грязный, много ошибок (outliers)
2. похоже, что RoBERT и BERT ошибаются на очень длинных текстах, пожно попробовать ToBERT https://arxiv.org/pdf/1910.10781.pdf
3. так же можно попробовать per-lang TF-IDF-based classic ML решения, может быть попробовать AutoML


### Что дальше?

1. Пробовать чистить данные, как-то идентифицировать outliers и выкинуть их из обучающей выборки, но не все, оставив как-то шум.
2. Пробовать ToBERT https://arxiv.org/pdf/1910.10781.pdf
3. Попробовать per-lang TF-IDF-based classic ML решения, может быть попробовать AutoML
4. перевести больше toxic сообщений на ES, PT, TR, IT, FR и даже RU, т.к. train data сильно скошена в сторону non-toxic комментариев
