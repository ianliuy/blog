# linux运维

1. 安装ipykernel

```bash
pip insatll ipykernel
```

2. 将ipykernel注册，命名

```bash
python -m ipykernel --user(optional) --name XXX
```

3. 相关命令：查看kernelspce里有哪些kernel

```bash
jupyter kernelspce list
```

4. 相关命令：删除kernel

```bash
jupyter kernelspec remove name
```

*reference*: https://ipython.readthedocs.io/en/latest/install/kernel_install.html

5. 相关命令：开jupyter server

```bash
jupyter notebook --ip=0.0.0.0(optional) --port=7777 --allow-root(optional) --no-browser
```

*reference*: https://jupyter.readthedocs.io/en/latest/running.html#starting-the-notebook-server

6. du -sh *查看文件大小

```bash
-s --summarize 总和
-h --human-readable 转化成K M G等单位易读
--time 显示目录下的修改时间
-S --separate-dirs 不计算子文件夹
```
example

```bash
root@XXX:~/XXX# du -sh --time *
406M	2020-02-19 07:38	finnews_online_XX
6.6M	2020-02-25 09:00	log
1.7G	2020-02-18 14:10	news_company_XX
4.6G	2020-03-11 02:25	spider
```

*reference*: https://linux.die.net/man/1/du

7. (不懂就写) for i in range(10)

> 结果是0-9的数字。也就是说10个数，不包括10。

