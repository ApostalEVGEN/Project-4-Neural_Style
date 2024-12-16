# Project-4-Neural_Style

## Оглавление  
[1. Описание проекта](.README.md#Описание-проекта)  
[2. Какой кейс решаем?](.README.md#Задачи)    
[3. Краткая информация о данных](.README.md#Результат)   
[4. Этапы работы над проектом](.README.md#Результат)   
[5. Результат](.README.md#Результат)   


### Описание проекта    
Перенос стиля является наиболее известной задачей машинного обучения. Во многом термины искусственный интеллект, глубокое обучение и перенос стиля (style-transfer) стали популярны благодаря приложению Prisma. Разработчики приложения одними из первых реализовали вычисления модели на мобильном устройстве. Тому было несколько причин:

Количество пользователей непрерывно росло. 
Приложение было бесплатным, серверы стоили денег, а также создали дефицит карточек у провайдера.
2016 год (трафик всё ещё стоит денег).
В целом, в индустрии разработки задачи переноса стиля используются редко. В основном такие задачи реализуются в приложениях для переносных устройств.

Сейчас мы можем наблюдать множество приложений с инференсом прямо на устройстве. Также появилось много способов оптимизации модели, чтобы ускорить приложение и сохранить пользователю немного заряда на устройстве, а разработчикам — денег.

:arrow_up:[к оглавлению](_)


### Какой кейс решаем?  

Ваша задача — обучить модель, оптимизировать её и импортировать веса в приложение.

Ваша основная задача — создать правдоподобный прототип работающей модели переноса стиля.

Цель — мобильное приложение или другой прототип, реализующий функцию, а также демонстрация его работы.

:arrow_up:[к оглавлению](.README.md#Оглавление)


### Краткая информация о данных

В нашем случае мы будем использовать произвольные изображения размером 256х256 и точно такое же изображения для стиля
при обучении модели. Модель будем обучать при помощи PyTorch на базе VGG-19. Используем Jupyter notebook, а также 
терминал Anaconda для создания страницы Flask и обернём нашу модель этим декоратором.

:arrow_up:[к оглавлению](.README.md#Оглавление)


### Этапы работы над проектом

Подбор базы изображений для контента и стиля
Написание кода и получение модели PyTorch
Написание кода для создания среды flask и запуска модели переноса стиля



:arrow_up:[к оглавлению](.README.md#Оглавление)

### Результат:  

Проведено обучение модели на большом количестве изображений (пришлось даже дублировать часть изображений по нескольку раз), получена вполне работоспособная модель. Много времени заняло задача для создания страницы в среде Flask. В конечном итоге 
модель работает и мы можем получать изображения с переносом стиля.


P.S. Очень много времени ушло на данный проект. Изначально пытался работать с неактивным репозиторие Magenta в системе Linux
Ubuntu, но как оказалось, работоспособных бтблиотек для этого репозиторя уже нет, а по времени все эти попытки заняли 
порядка 2-2,5 месяца. Затем решил сделать модель на PyTorch, не сразу конечно, также пришлось для этой цели сменить видеокарту, но модель была обучена (из-за неудач хотел уже создавать модель на TensorFlow), ещё порядка 1,5 месяца. Заткм попытка работать с Kotlin для создания Android-приложения на базе модели, но увы, 2 необходимые библиотеки в приложении 
Android Studio невозможно было установить (связано с действующими на данный момент санкции в отношении нашей страны), а это 
ещё около месяца времени впустую. В итоге была создана Flask-страница, разумеется не с первого раза, но получилась 
работоспособная версия для работы с изображениями.
