# 字符串中的Unicode character property和Unicode equivalence
关于这个问题你可以看一下字符的类别, `unicodedata.category(unichr)`可以返回 str 的常规类别([general category]( https://en.wikipedia.org/wiki/Unicode_character_property)). 比如中文或一些外文的的声调符号的 category 是`Mn`.

`unicodedata.normalize(form, unistr)`可以返回 Unicode 字符串 unistr 的普通形式“form”, 有`NFC`, `NFKC`, `NFD`和`NFKD`等, 详细解释见[unicode equivalence]( https://en.wikipedia.org/wiki/Unicode_equivalence)

比如一段代码:

```python
>>> s = 'Ślusàrski'
>>> print(s)
'Ślusàrski'
```

现在是 NFC composition 形式, 也就是'Ś' = 'Ś'

转换成 NFD decomposition 形式:

```python
>>> print(ascii(unicodedata.normalize('NFD','Ślusàrski')))
'S\u0301lusa\u0300rski'
```

'Ś' = 'S\u0301'

因此可以写一个 unicode_to_ascii 的函数:

```python
>>> s = 'Ślusàrski'
>>> def unicode_to_ascii(s):
>>> return ''.join(c for c in unicodedata.normalize('NFD', s)
>>> if unicodedata.category(c) != 'Mn')
>>> print(unicode_to_ascii(s))
Slusarski
```