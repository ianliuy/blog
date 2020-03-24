# screen增删查改

## 增加

增加新的screen屏幕

```bash
screen -S name
```
![image.png](https://tva1.sinaimg.cn/large/006xRaCrly1gd55j2drdqj31hc07d3zv.jpg)
连接detached screen

```
screen -r name/id
```
![image.png](https://tva1.sinaimg.cn/large/006xRaCrly1gd55ij3o01j31gl0cd76h.jpg)
连接attached screen

```
screen -D  -r name/id
```
![image.png](https://tva1.sinaimg.cn/large/006xRaCrly1gd55h487boj30x004eglq.jpg)

## 删除

从现有的screen屏幕中分离

```bash
Ctrl + a + d
```

删除某个screen屏幕

```
screen -S name -X quit
```
![image.png](https://tva1.sinaimg.cn/large/006xRaCrly1gd55fv9la1j31gu07pmyd.jpg)

或者，激活某个screen后exit

```
screen -r name/id
exit
```

## 查

查看所有screen

```
screen -ls
```
![](https://tva1.sinaimg.cn/large/006xRaCrly1gd553usxfuj31hd0hc76o.jpg)

## 改

session改名

```
screen -S old_session_name -X sessionname new_session_name
```
![image.png](https://tva1.sinaimg.cn/large/006xRaCrly1gd55p34tjkj31it0eh76t.jpg)

*reference*: https://www.gnu.org/software/screen/manual/screen.html