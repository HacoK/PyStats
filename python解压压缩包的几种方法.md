# 转：python解压压缩包的几种方法

这里讨论使用Python解压例如以下五种压缩文件：

.gz .tar  .tgz .zip .rar

## 简单介绍

gz： 即gzip。通常仅仅能压缩一个文件。与tar结合起来就能够实现先打包，再压缩。

tar： linux系统下的打包工具。仅仅打包。不压缩

tgz：即tar.gz。先用tar打包，然后再用gz压缩得到的文件

zip： 不同于gzip。尽管使用相似的算法，能够打包压缩多个文件。只是分别压缩文件。压缩率低于tar。

rar：打包压缩文件。最初用于DOS，基于window操作系统。

压缩率比zip高，但速度慢。随机訪问的速度也慢。

关于zip于rar之间的各种比較。可见： 

<http://www.comicer.com/stronghorse/water/software/ziprar.htm>

## gz

因为gz一般仅仅压缩一个文件，全部常与其它打包工具一起工作。比方能够先用tar打包为XXX.tar,然后在压缩为XXX.tar.gz

解压gz，事实上就是读出当中的单一文件，Python方法例如以下：

 

## tar

XXX.tar.gz解压后得到XXX.tar，还要进一步解压出来。

*注：tgz与tar.gz是同样的格式，老版本号DOS扩展名最多三个字符，故用tgz表示。

因为这里有多个文件，我们先读取全部文件名称。然后解压。例如以下：

 

*注：tgz文件与tar文件同样的解压方法。

 

## zip

与tar类似，先读取多个文件名称，然后解压。例如以下：

 

## rar

由于rar通常为window下使用，须要额外的Python包rarfile。

可用地址： <http://sourceforge.net/projects/rarfile.berlios/files/rarfile-2.4.tar.gz/download>

解压到Python安装文件夹的/Scripts/文件夹下，在当前窗体打开命令行,

输入Python setup.py install

安装完毕。

 

 

tar打包

在写打包代码的过程中，使用tar.add()添加文件时，会把文件本身的路径也加进去，加上arcname就能依据自己的命名规则将文件添加tar包

打包代码：

1. \#!/usr/bin/env /usr/local/bin/python  
2.  # encoding: utf-8  
3.  import tarfile  
4.  import os  
5.  import time  
6.   
7.  start = time.time()  
8.  tar=tarfile.open('/path/to/your.tar,'w')  
9.  for root,dir,files in os.walk('/path/to/dir/'):  
10. ​         for file in files:  
11. ​                 fullpath=os.path.join(root,file)  
12. ​                 tar.add(fullpath,arcname=file)  
13.  tar.close()  
14.  print time.time()-start  

 

在打包的过程中能够设置压缩规则,如想要以gz压缩的格式打包

tar=tarfile.open('/path/to/your.tar.gz','w:gz')

其它格式例如以下表：

| 'r' or 'r:*' | Open for reading with transparent compression (recommended). |
| ------------ | ------------------------------------------------------------ |
| 'r:'         | Open for reading exclusively without compression.            |
| 'r:gz'       | Open for reading with gzip compression.                      |
| 'r:bz2'      | Open for reading with bzip2 compression.                     |
| 'a' or 'a:'  | Open for appending with no compression. The file is created if it does not exist. |
| 'w' or 'w:'  | Open for uncompressed writing.                               |
| 'w:gz'       | Open for gzip compressed writing.                            |
| 'w:bz2'      | Open for bzip2 compressed writing.                           |

 

tar解包

tar解包也能够依据不同压缩格式来解压。

1. \#!/usr/bin/env /usr/local/bin/python  
2.  # encoding: utf-8  
3.  import tarfile  
4.  import time  
5. 
6.  start = time.time()  
7.  t = tarfile.open("/path/to/your.tar", "r:")  
8.  t.extractall(path = '/path/to/extractdir/')  
9.  t.close()  
10.  print time.time()-start  

 

上面的代码是解压全部的，也能够挨个起做不同的处理，但要假设tar包内文件过多，小心内存哦~

```
1. tar = tarfile.open(filename, 'r:gz')  
2. for tar_info in tar:  
3. file = tar.extractfile(tar_info)  
4. do_something_with(file)  
```