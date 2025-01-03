# Prediction of the secondary structure of a protein


Данный проект является промежуточной целью. Конечная цель - предсказание третичной структуры белка по его аминокислотной последовательности. 

Пройдемся по структуре задачи. Аминокислотная последовательность - цепочка аминокислот, для нашего проекта можно считать, что входные данные - это строка с алфавитом в 25 букв. Cами по себе они  не обаладают какими-то особыми биолоигическими или физическими свойствами, полезными для организма. НО Гораздо интереснее становится, когда они благодаря различным связям (водородные связи, ионные, окислительно-восстановительные) и взаимодействиями (ваан-дер-вальсова сила) сжимаются и скручиваются, образуя некоторые структуры. Первыми такими структурами являются вторичные (а первичными называют саму аминокислотную последовательность). Они бывают трех видов: альфа-спирали, бета-слои и просто какие-то неопределенные участки. Эти структуры тоже в отдельности не несут каких-то очень полезных свойств - являются промежуточным сжатием. А уже потом, вся эта масса вторичных структур одного белка приобретает свою пространственную форму - третичная структура. А несколько таких третичных структур могу вместе образовывать четвертичную. Но это уже конечная цель проекта (по сути, все еще открытая задача - предсказание третичной структуры белка). И вот уже эти структуры является невероятно эффективными по действию и разнообразными по свойствам веществами. Тут и ферменты - супер эффективные катализаторы, ускоряющие проведение реакций в тысячи раз. И белки переносчики - к примеру гемоглобин, транспортировщих кислорода. 

Но вернемся ко вторичной структуре. Каждая аминокислота может принадлежать только к одному участку вторичной структуры - либо к альфа-спирали, либо к бета-слою, либо просто к неопределнному учатску в белке. На языке машинного обучения - нам надо классифицировать каждую аминокислоту. То есть вход - строка из n символов со словарем 25, выход - строка из n символов со словарем 3. Естественно белки могут состоять из аминоксилотных цепочек различной длины и даже из нескольких таких цепочек (в случае если конечная цель - четвертичная структура). В нашей задаче если у ебкла несколько цепочеку - будем просто разбивать их на разные примеры. А в случае переменной длины - добавлять паддингами до размера макисмальной цепочки.

Сразу скажу - задача оказалась легче чем я думал. Легче именно в плане того, что не пришлось создавать вместо модели - монстра, чтобы получить достойное качество. Можно было просто сделать достаточно глубокую и более менее разумную модель из одномерных сверток с нелинейностями и нормализациями и задача решена. Поэтому, раз не получилось пойти вглубь, я пошел вширь и старался опробовато как можно больше методов и подходов.


Первая наша задача - собрать свою базу данных. Это оказалось достаточно сложно, потому что готовых датасетов для этой задачи просто нет(я не нашел). Сами аминокислотные последовательности брались из базы данных PDB непосредственным парсингом. Потом оказалось, что в той же базе данных есть далеко не все последовательности вторичных структур. Поэтому пришлось воспользоваться доплнительной программой для их получения(данная программа, по сути, занимается решением нашей задачи, но нам нужны свои модели и свое исследование!). Весь код по созданию баззы данных содержится в . 

Переходим к обработке даннхы и токенизации. Обрабатывать нечего - у нас есть строка из символов и все они нам нужны. В качестве токена было решено использовать n-граммы. А именно триграмма и кватрогаммы (на графиках эксперименты только с кватрограммами - разницы между ними и триграммами выявлено не было). Решение использовать их довольно банальное - у нас маленьких словарь и вместо того, чтобы предсказывать каждой аминоксилоте один из трех классов, будем предсказывать каждой n-грамме класс. Таким образом мы очень сильно расширяем словарь - 159111 уникальных n-грамм. Здесь стоит упомянуть, что были выкинуты все n-граммы, частота встречаемости которых ниже 3. Ну а дальше просто готовим нашу выборку из n-грамм. Ознакомиться с этим кодом можно в файле - tokenization.

Следом идет создание эмбеддингов. Было использовано два способа - обучение с нуля и файтьюн. С обучением с нуля все понятно - просто эмбеддинг слой в модели. А вот в качестве предобученных были использована два подхода - GloVE и protBERT. GloVE был реализован с нуля, а вот protBERT - взят из гитхаба ( ). Как и гласит название - это специльно обученный BERT на аминокислотах. Весь код по созданию эмбеддингов соердится в папке Embeddings.

По созданию DataSet и DataLoader сказать особо нечего. В датасете готовим данные, индексируем. В даталодере с помощью collat_fn добиваем все входные данные до единого размера с помощью паддингов. Код содержится в файле dataset_and_dataloader.

Переходи к самому вкусному - к моделям.

Удивительно, но LSTM сама по себе решает эту задачу недостаточно хорошо. 
После перешел на трансформеры но быстро уперся в память видеокарты и бросил эту идею. И решил посмотреть что есть сейчас конкурентного у трансформеров и дошел до двух интересных статей - про SSM (Mamb-у мы реализовывали на параллельном курсе по deep learning, поэтому я не стал вносить её сюда), xLSTM (супер модификация от авторов оригинальной LSTM - имеет два блока - sLSTM - просто улучшения архитекутры LSTM и mLSTM - с матричной памятью, что позволяет её распарелливать и выводит на один уроввень с трансформерами) и Learn at Test Time: RNNs with Expressive Hidden States (https://arxiv.org/abs/2407.04620) - очень интересная вещь. В первую очередь мне понравилась идея с обучаемыми скрытыми состояниями. То есть скрытое состояние это больше не просто матрица с хранением информации, а практические отдельная моделька, которая обучается вместе с общей сетью. Именно эту идею из статьи я и постарался реализовать. И вроде как получилось - в первую очередь  модель получилась гораздо быстрее, чем LSTM, и качество выдает чуть лучше (тут конечно стоит упомянуть - что в статье говорится, что по качеству она уделывает трансформеры, но моей реализации до такого очень далеко). Описать алгоритм я уже не успею, но добвалю его сюда позже (так как планирую развивать работу над этой задачей и дальше).

Ну и решение, которое порвало все другие методы .... CNN. Да, одномерные свертки просто взлетели очень высоко в этой задаче. Я достаточно долго провозился здесь с ними и дошло даже до того, что я переписал ResNet на одномерный случай (естественно, его мини версию). Особого прироста не дало, но тут видимо я уперся в потолок возможностей CNN для этой задачи.


А дальше уже пошли попытки выбить 100% качество с помощью объединения методов. ResNet + LSTM, ResNet + TTT, ResNet + Transformer Encoder.

Лучшим по качеству оказалась комбинация предобученые эмбеддинги на protBERT, последующее их дооубение + ResNet1d + TTT - выбило качество чуть выше 95% на тесте и 97,5% на обучении.

Вот здесь можно посмотреть на визуализацию предсказания. Как видно, вся последовательность предсказывается верно.
![Визуализация ](https://github.com/foxsly5425/-Prediction-of-the-secondary-structure-of-a-protein/blob/main/image/visualization.jpg)

Помимо уже заданной выше формулировки задачи, я попробовал применить seq2seq структуру для решения этой задачи. Хоть она и избыточна, но попробовать было интересно. Главная проблема подхода - вместимость видеокарты. На графиках это изображено не будет, но мне удалось обучить только половину одной эпохи на полноценном трансформере(с энкодером и декодером) - обрезал данные в 4 раза, батч размером =8, всего две головки и по два слоя и в энкодере и в декодере. За половину эпохи accuracy дошла до 80%. 

Также внедрил seq2seq из 2-ой домашки про перевод - с lstm и механизмом внимания, но тут тоже плохо пошло - всего 2 эпохи влезло.

Не усспел собрать все графики в кучу, потом вывел всего несколько для доказательства работы моделей

![графики ](https://github.com/foxsly5425/-Prediction-of-the-secondary-structure-of-a-protein/blob/main/image/graphics.jpg)
