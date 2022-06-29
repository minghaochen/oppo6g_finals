# OPPO 6G AI based Channel Modeling and Generating

## 队伍

队名：刺客五六七

排名：Rank 6

线上成绩：0.70555377

## 环境

load docker镜像

```
docker load -i oppo6g.tar
```

启动容器并进行文件夹映射

```
docker run -it -v $path$ :/workfolder:ro oppo6g
```

注：path为oppo6g_finals的前缀地址

## 指令

数据集1

```
python /workfolder/oppo6g_finals/data/train_code/qae_main.py --dataset 1
```

数据集2

```
python /workfolder/oppo6g_finals/data/train_code/qae_main.py --dataset 2
```

Note：线下显卡3070（8g），两个数据集在5小时内训练完毕

## 提交

user_data里包含最佳模型

```
generator_1.pth.tar
generator_2.pth.tar
```

以及线上推理文件

```
generatorFun.py
```



