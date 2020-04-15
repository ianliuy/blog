# python中iterator和generator

The iterator objects themselves are required to support the following two methods, which together form the [iterator protocol](https://docs.python.org/3/library/stdtypes.html#iterator-types):

iterator.__iter__()
Return the iterator object itself. This is required to allow both containers and iterators to be used with the **for** and in **statements**. 

iterator.__next__()
Return the next item from the container. If there are no further items, raise the **StopIteration** exception.

Python defines several iterator objects to support iteration over general and specific sequence types, dictionaries, and other more specialized forms. The *specific types* are not important beyond their implementation of the iterator protocol.

Once an iterator’s __next__() method raises **StopIteration**, it must continue to do so on subsequent calls. Implementations that do not obey this property are deemed broken.