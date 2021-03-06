Постановка задачи: На примерах Единого Государственного Экзамена решить проблему лексической омонимии.

В качестве данных использовался датасет соревнования сбербанка https://contest.ai-journey.ru/ru/competition

Начну с того, что у меня не удалось воспроизвести решение победителей и получить их скор на тесте. У меня он получался 0.4 вместо заявленных
0.65, поэтому я решил самостоятельно экспериментировать с решением задачи, пробуя разные методы, даже заведомо зная, что они дадут, вероятнее всего, не самый лучший результат
(ведь главное для меня было получить опыт с BERT:))
Ниже привожу основные эксперименты.

1.Изначально я попробовал разбить задание на текст, варианты ответов, слово, вопрос.
Затем получил эмбеддинги берта путем конкатенации последних 4 слоев для токена CLS (тексты и варианты ответа).
Размерность каждого вектора получилась 3072. Далее я попробовал найти меру похожести каждого варианта ответа и текста через косинусное расстояние и взять самый высокий скор балл.
Результат - 0.36 accuracy

2.Также разбил задание на текст, варианты ответов, слово, вопрос.
Затем получил эмбеддинги берта 9 и 11 слоев по вариантам ответа и предложению, в котором содержится ключевое слово.(эмбеддинги всех токенов, не только CLS), улучшение на 5%
Также пробовал различные варианты эмбеддингов, но 9 и 11 дали наилучший результат
Далее я пробовал чистить дополнительно текст от разных вставок и символов (цифр предложений, пропущенный текст в виде <...>). Для пункта 1 никакого результата не дает,
а вот в эмбеддингах предложений <...> играет некую роль, т.к результат становится хуже.

3.Попробовал вытащить contextual word embedding для слова, которое необходимо определить. Затем также искал косинусную близость между этим
эмбеддингом, и эмбеддингом ответа. Результат улучшился до 46%.
Если делать среднее эмбеддингов 9 и 11 слоев, результат вырастает до 48.57%
Также пробовал посмотреть взаимосвязь между "предсказаниями" разных варинатов эмбеддингов, с целью выяснить, есть ли смысл пытаться
взвешивать результаты косинусной близости между методами (своего рода ансамбль). Однако результаты косинусной близости оказывались сильно  
коррелированными.


4.Далее попробовал заменить multilingual bert на RUBERT. Сначала получил результат хуже относительно тех же конфигураций слоев(40%), однако затем поигрался со слоями, сделал 
concat 0-11 слоев, получил результат значительно лучше - 54.28%.
Объединение этих же слоев в multilingual bert не изменяет результат (48.57%)

Выводы: задача оказалась достаточно сложной, по моему мнению в ней присутствует большая доля стохастики, так как от изменения пары символов (см. пункт 2), могло меняться качество предсказаний.
Поэтому один из выводов, и, возможно, вариант дальнейшего действия - чистка/обработка данных (хотя пока плохо представляю, что в этом направлении еще можно сделать).
Также интересно заметить, что для RuBert и multilingual BERT были оптимальны разные комбинации hidden layers. + всегда стоит пробовать разновидности BERT (RuBert в моем случае), даже если говорят, что он обычно не выстреливает:)
Соответственно в будущем хотелось бы попробовать использовать другие трансформеры, например, xlm-roberta.