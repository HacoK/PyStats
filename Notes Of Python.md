# Notes Of Python

## 1、open中的newline参数

On input,if newline is None, universal newlines mode is enabled. Lines in the input can end in '\n', '\r', or '\r\n', and these are translated into '\n' before being returned to the caller. If it is '', universal newline mode is enabled, but line endings are returned to the caller untranslated. If it has any of the other legal values, input lines are only terminated by the given string, and the line ending is returned to the caller untranslated.

On output,if newline is None, any '\n' characters written are translated to the system default line separator,os.linesep. If newline is '', no translation takes place. If new line is any of the other legal values, any '\n' characters written are translated to the given string.
PS：csv标准库中的writerow在写入文件时会加入'\r\n'作为换行符 

## 2、str()和repr()

The str() function is meant to return representations of values which are fairly

human-readable, while repr() is meant to generate representations which can be read by

the interpreter (or will force a SyntaxError if there is not equivalent syntax). For

objects which don't have a particular representation for human consumption, str() will

return the same value as repr(). Many values, such as numbers or structures like lists

and dictionaries, have the same representation using either function. Strings and

floating point numbers, in particular, have two distinct representations.

### 3、Python zip() 函数

### 描述

**zip()** 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。

如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。

### 语法

zip 语法：

zip([iterable, ...])

参数说明：
iterabl -- 一个或多个迭代器;

### 返回值

返回元组列表。

### 实例

以下实例展示了 zip 的使用方法：

\>>>a = [1,2,3]>>> b = [4,5,6]>>> c = [4,5,6,7,8]>>> zipped = zip(a,b)     # 打包为元组的列表[(1, 4), (2, 5), (3, 6)]>>> zip(a,c)              # 元素个数与最短的列表一致[(1, 4), (2, 5), (3, 6)]>>> zip(*zipped)          # 与 zip 相反，可理解为解压，返回二维矩阵式[(1, 2, 3), (4, 5, 6)] 

## 4、Python translate()方法

### 描述

Python translate() 方法根据参数table给出的表(包含 256 个字符)转换字符串的字符, 要过滤掉的字符放到 del 参数中。

### 语法

translate()方法语法：

str.translate(table[,deletechars]);

### 参数

- table -- 翻译表，翻译表是通过maketrans方法转换而来。
- deletechars -- 字符串中要过滤的字符列表。

### 返回值

返回翻译后的字符串。

### 实例

以下实例展示了 translate()函数的使用方法：

\#!/usr/bin/python

from string import maketrans \# 引用 maketrans 函数。

intab = "aeiou"

outtab = "12345"

trantab = maketrans(intab, outtab)

str = "this is string example....wow!!!"

print str.translate(trantab);

以上实例输出结果如下：

th3s 3s str3ng 2x1mpl2....w4w!!!

以上实例去除字符串中的 'x' 和 'm' 字符：

\#!/usr/bin/python

from string import maketrans 

\# Required to call maketrans function.

intab = "aeiou"

outtab = "12345"

trantab = maketrans(intab, outtab)

str = "this is string example....wow!!!";

print str.translate(trantab, 'xm');

以上实例输出结果：

th3s 3s str3ng 21pl2....w4w!!!

## 5、Operations of List and Dict in Python

### List:

L.append(var)   #追加元素

L.insert(index,var)

L.pop(var)      #返回最后一个元素，并从list中删除之

L.remove(var)   #删除第一次出现的该元素

L.count(var)    #该元素在列表中出现的个数

L.index(var)    #该元素的位置,无则抛异常

L.extend(list)  #追加list，即合并list到L上

L.sort()        #排序

L.reverse()     #倒序

list 操作符:,+,*，关键字del

a[1:]      #片段操作符，用于子list的提取

[1,2]+[3,4] #为[1,2,3,4]。同extend()

[2]*4      #为[2,2,2,2]

del L[1]   #删除指定下标的元素

del L[1:3] #删除指定下标范围的元素

list的复制

L1 = L     #L1为L的别名，用C来说就是指针地址相同，对L1操作即对L操作。函数参数就是这样传递的

L1 = L[:]  #L1为L的克隆，即另一个拷贝。

### **Dict:**

1、如何访问字典中的值？

adict[key] 形式返回键key对应的值value，如果key不在字典中会引发一个KeyError。

2、如何检查key是否在字典中？

a、has_key()方法 形如：adict.haskey(‘name') 有–>True，无–>False

b、in 、not in   形如：'name'inadict      有–>True，无–>False

3、如何更新字典？

a、添加一个数据项（新元素）或键值对

adict[new_key] = value 形式添加一个项

b、更新一个数据项（元素）或键值对

adict[old_key] = new_value

c、删除一个数据项（元素）或键值对

del adict[key] 删除键key的项 / deladict 删除整个字典

adict.pop(key) 删除键key的项并返回key对应的 value值

 

1、adict.keys() 返回一个包含字典所有KEY的列表；

2、adict.values() 返回一个包含字典所有value的列表；

3、adict.items() 返回一个包含所有（键，值）元祖的列表；

4、adict.clear() 删除字典中的所有项或元素；

5、adict.copy() 返回一个字典浅拷贝的副本；

6、adict.fromkeys(seq,val=None) 创建并返回一个新字典，以seq中的元素做该字典的键，val做该字典中所有键对应的初始值（默认为None）；

7、adict.get(key,default = None) 返回字典中key对应的值，若key不存在字典中，则返回default的值（default默认为None）；

8、adict.has_key(key) 如果key在字典中，返回True，否则返回False。 现在用 in 、 not in；

9、adict.iteritems()、adict.iterkeys()、adict.itervalues()与它们对应的非迭代方法一样，不同的是它们返回一个迭代子，而不是一个列表；

10、adict.pop(key[,default])和get方法相似。如果字典中存在key，删除并返回key对应的vuale；如果key不存在，且没有给出default的值，则引发keyerror异常；

11、adict.setdefault(key,default=None) 和set()方法相似，但如果字典中不存在Key键，由 adict[key] = default 为它赋值；

12、adict.update(bdict) 将字典bdict的键值对添加到字典adict中。

遍历key    for keyinadict.keys():print key

遍历value  for valueinadict.values(): print value

遍历项     for iteminadict.items():print item