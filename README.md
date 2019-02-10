# Disaster Response Pipeline Project

#####  GitHub repository：git@github.com:flysaint/My_Poject_Disaster_Response_Pipeline.git

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### 大致思路

#### 第一部分——清洗数据。在data/ 里，运行process_data.py
1. 载入数据。 load_data，载入 messages 和 categories数据，并进行合并
2. 清洗数据。主要是讲 categories进行处理，里面的数据格式是： “字段名1-数字;字段名2-数字;字段名3-数字”。
2.1 用";" 将每段字段名和数字截取出来.
2.2 产生新的列名。使用第一行，将截取出来的字段名-数字，再次用-进行截取。使用字段名作为新的列名
2.3 将截取产生的字符型数字转换为数值型。
2.4 删除重复数据。
3. 保存数据。保存数据到执行的数据库和表名

#### 第二部分——训练模型。在 models/train_classifier.py
1. 载入数据。就是第一部分中保存的数据。其中数值数据，作为目标值Y，使用message作为X值。
2. 训练模型。
2.1 建立 tokenzie。定义文本数据的ETL方法。
2.2 建立pipeline。定义tfidf，随机森林多目标分类器
2.3 定义优化参数。使用GridSearchCV进行搜索调优。
3. 模型评估。训练和预测模型，输出模型的训练结果。
4. 保存模型。

#### 第三部分——使用Flask展示数据。在app/run.py里。
1. 载入数据和模型。
2. 依据Flask框架。查询和展示数据。
2.1 定义go函数。功能，发送请求，返回结果。
2.2 定义index函数。依据返回的数据，展示数据

#### 其他部分
templates。Flask的数据展示界面

