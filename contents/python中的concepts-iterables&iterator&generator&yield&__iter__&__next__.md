# python中的concepts-iterable&iterator&generator&yield&\_\_iter\_\_&\_\_next\_\_

## iterables

*reference*: https://www.pythonlikeyoumeanit.com/Module2_EssentialsOfPython/Iterables.html#Iterables

*definition*:
 > *iterable*是一个python [Abstract Base Classes](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable), 有\_\_iter\_\_()方法的类都可以被称为iterable. 并且官方文档说有\_\_getitem\_\_() method的class不一定是iterable.