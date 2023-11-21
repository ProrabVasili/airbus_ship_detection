# Airbus Ship Detection Challenge

[![Airbus_kaggle](https://github.com/ProrabVasili/airbus_ship_detection/assets/90131636/244baac7-d782-4ef3-b969-f0e80772d378)](https://www.kaggle.com/competitions/airbus-ship-detection/overview)

_Цей проект присвячений semantic segmentation щодо детекту кораблів. Уся "еквілібристика" виконувалась у Colab'i (через слабкий пристрій), але там було обмеження на пам'ять та оперативну пам'ять, а тому датасет був зменшений щодо тренування до 10k.
Для semantic segmentation використовувалась U-Net model.
Сподіваюсь, що із покращенням своїх технологій колись повернусь до даного проекту, аби навчити модель на всіх даних._



**_Міні-EDA_**

Спочатку було проаналізовано стартовий csv на кількість пропущених даних, було побудовано "розподіл" між кількістю зображень та кораблів на них, а також самі демонстраційні варіанти між зображень, його маскою та виявленими областями (для цього було реалізовано було згруповано дані за id, а потім конвертовано маски завдяки rle_decoder)
![Bin](https://github.com/ProrabVasili/airbus_ship_detection/assets/90131636/abd2c865-2e3c-4483-99c1-2048fcacaf5b)


- Приклад із невиявленими кораблями
![Without_ships](https://github.com/ProrabVasili/airbus_ship_detection/assets/90131636/7e791cc1-b390-4791-ab96-a18a4cebda44)


 - Приклад із виявленими кораблями
![With_ships](https://github.com/ProrabVasili/airbus_ship_detection/assets/90131636/0a801e50-db6a-4d96-bde1-6c7c747f254e)



**_Навчання_**

Для самої реалізації U-Net було обрано Tensorflow. За метрику було обрано dice coefficient:
![Dice](https://github.com/ProrabVasili/airbus_ship_detection/assets/90131636/65a0816a-b5a6-4c0c-a3b2-b3404dad88d1)

За Loss було обрано комбінацію Dice loss та Binary Crossentropy

У підсумку вийшла така модель:
![Coeff_plot](https://github.com/ProrabVasili/airbus_ship_detection/assets/90131636/657310bb-7cc1-4c74-8172-6c56436edd17)


![Loss_plot](https://github.com/ProrabVasili/airbus_ship_detection/assets/90131636/8d08fd21-c104-4c3d-966a-ada5cc8b8766)


Як можемо бачити, то є куди покращуватись моделі.
Ось приклад її предікту:
![predict](https://github.com/ProrabVasili/airbus_ship_detection/assets/90131636/2e0cb881-5505-41cd-a5f9-7ef993ea4f4e)


**_Користування_**
- Спочатку Вам треба встановити/оновити бібліотеки, що вказані у requierements.txt.
- Якщо Ви хочете навчити свою модель, то в model_training.py треба змінити шляхи для train_data та test_data.
- Якщо Ви хочете предіктити на існуючій моделі, то в model_inference.py треба змінити шлях папки із predict_data
- Для тих, кому зручний Jupiter, то було надано ще і model.ipynb
