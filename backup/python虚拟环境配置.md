
## virtualenv
* 使用virtualenv创建虚拟环境，名称：my_ml_env，表示我的机器学习环境。
### 安装所需的包
* 管理员身份运行下列命令
    * pip install virtualenv
    * pip install virtualenvwrapper-win

### 配置环境变量
* 确保你的系统环境变量中包含 Python 的 Scripts 文件夹路径，并配置 WORKON_HOME。
    * 打开 系统属性。
    * 进入 高级 > 环境变量。
    * 在 系统变量 中找到 Path，并添加 Python 的 Scripts 文件夹路径（例如：C:\Python39\Scripts）。
    * 新建一个名为 WORKON_HOME 的变量，值为你希望创建虚拟环境的路径（例如：C:\Users\YourUsername\Envs）。

### 常用命令
virtualenvwrapper-win 提供了一些常用命令来管理虚拟环境：
* 创建虚拟环境：mkvirtualenv myenv / mkvirtualenv --python=python3.6 myenv
* 列出所有虚拟环境：lsvirtualenv 或 workon
* 激活虚拟环境：workon myenv
* 退出虚拟环境：deactivate
* 删除虚拟环境：rmvirtualenv myenv
* 查看虚拟环境中的安装包：pip list

### 虚拟环境添加到jupyter
* 在虚拟环境中执行安装所需包：
    * pip install jupyter
    * pip install ipykernel
    * python -m ipykernel install --name 虚环境名称 --display-name 虚环境名称 --user

## conda
* 使用conda创建虚拟环境，名称：conda_ml_env，表示conda机器学习环境。
* [Anaconda3+Pycharm 搭建深度学习环境；安装不同框架；配置两个环境；Anaconda配置国内镜像源](https://blog.csdn.net/2302_76846184/article/details/138009420)
### 在conda中创建环境步骤
    * 打开Anaconda Prompt(Anaconda3)
    * 执行下面命令，可以指定基于特定版本的python环境；
        * conda create --name wuyi python=3.7.1
    * 激活环境：conda activate conda_ml_env

### 常用命令
* conda create --name YourName python=3.8.8  # 创建环境 YourName 是你需要命名的环境，比如可以命名为yolov5，后面的python=python版本
* conda activate YourName  # 激活环境你的环境
* cd D:/DeepLearning/  # cd到有DeepLearning的目录   cd是切换目录命令
* pip install -r requirements.txt  # 安装项目需要的所有包
* conda env list  # 查看所有环境
* conda list  # 查看所有包
* conda remove -n YourEnvName --all  # 删除虚拟环境

### 虚拟环境添加到jupyter
* 虚拟环境中安装所需要的包
    * pip install jupyter
    * pip install ipykernel
    * python -m ipykernel install --name 虚环境名称 --display-name 虚环境名称 --user

## 删除jupyter中的虚拟环境
* cmd查看所有的环境 jupyter kernelspec list
* 删除环境 jupyter kernelspec remove myenv

