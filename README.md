Perfect Draco Summoning Circle
====

Generate beautiful drop layout for Draco Summoning Circle. (know as 最大火力配置 in Japan)

## Demo
Output example...
####31-11
![4390305789439C6L3W0](https://github.com/gumboshi/pzdr_saidai/blob/master/file/4390305789439C6L3W0.png)

####25-17
![25-17ID463222794863C11L0W6.png](https://github.com/gumboshi/pzdr_saidai/blob/master/file/25-17ID463222794863C11L0W6.png)

These images are generated by ID2table.jar.


## Expected simulation time 

#### Full simulation for 5x6 

GPU(Nvidia K20) < 3 sec.

CPU(6core Xeon) < 30 sec.

CPU(1core Xeon) < 2 min.

#### Full simulation for 6x7 

GPU(Nvidia K20) < 2.5 hour.

CPU(6core Xeon) < 2 day.

CPU(1core Xeon) < 1 week.


## Similar software
For 5x6 table, there is really sophisticated online service: http://full-combo.herokuapp.com/position

In contrast, this program supports 4x5, 5x6, and 6x7 tables. (It seems that not implemented yet for US version.)



## Requirement
If you want to use GPU version, CUDA toolkits (https://developer.nvidia.com/cuda-downloads) are required.


## Install

To use GPU, 
```shell
$ git clone https://github.com/gumboshi/pzdr_saidai.git
$ cd pzdr_saidai
$ make gpu
```
(Optional) If you know the number of SMX of your GPU, it is better for your system.
```shell
$ make gpu NUM_BLOCK=(4 * number of SMX)
```
You don't need gpu version,
```shell
$ git clone https://github.com/gumboshi/pzdr_saidai.git
$ cd pzdr_saidai
$ make
```

## Usage

#### Change table size
To use 4x5 table
```
$ ./pzdr_saidai.exe -small
```
To use 5x6 table (default)
```
$ ./pzdr_saidai.exe -normal
```
To use 6x7 table
```
$ ./pzdr_saidai.exe -big
```

#### Change Leader Skill
Heros (ex. Perseus, Pandora)
```
$ ./pzdr_saidai.exe -hero
```
Lakshmi and Parvati
```
$ ./pzdr_saidai.exe -laku (or -paru)
```
Krishna (Ultimate)
```
$ ./pzdr_saidai.exe -krishna
```
Now available: -hero, -laku(-paru), -krishna, -sonia 

#### Change the number of awoken skills
9 Enhanced ATT, 2 Two-pronged ATT
```
$ ./pzdr_saidai.exe -l 9 -w 2
```
#### Change the number of Orbs
If you specify -s and -e like following, simulation executes 10-20, 11-19, 12-18, 13-17, and 14-16
```
$ ./pzdr_saidai.exe -s 10 -e 14 -normal
```


## Output 

Default output form is like following, 
```
13-17, line 0, way 0
rank,tableID      ,power
   0,     36923626,    12.750
   1,     36924130,    12.750
   2,     45312202,    12.750
   3,     49049346,    12.750
   4,     49311426,    12.750
   5,     74074853,    12.750
   6,     74075112,    12.750
   7,     74119464,    12.750
   .          .          .
   .	      .		 .
   .	      .		 .
  99,    116238594,    12.000
```
tableID corresponds each table. To convert the ID to table, see Usage of visualization tool.

If you specify -ave, 10000 time simulation will execute to evaluate the influence of Orbs which fall from the outside of the table.
```
$ ./pzdr_saidai.exe -ave
```
And also output will be change like following, 
```
13-17, line 0, way 0
rank,tableID      ,power     ,ave power ,min power ,ave combo ,min combo
   0,     36923626,    12.750,    14.977,     8.125,    10.286,     7
   1,     36924130,    12.750,    15.045,     8.125,    10.329,     7
   2,     45312202,    12.750,    14.996,     8.125,    10.288,     7
   3,     49049346,    12.750,    15.094,    12.750,    10.250,     9
   4,     49311426,    12.750,    15.022,    12.750,    10.231,     9
   5,     74074853,    12.750,    12.787,     4.500,     9.490,     6
   6,     74075112,    12.750,    14.580,     8.938,    10.208,     8
   7,     74119464,    12.750,    14.530,     8.938,    10.170,     8
   .          .          .          .          .          .         .
   .          .          .          .          .          .         .
   .          .          .          .          .          .         .
  99,    116238594,    12.000,    14.087,     7.500,    10.160,     7
```
ave power... average of the 10000 time simulation

min power... minimum ATT of the 10000 time simulation

ave combo... average number of combo of the 10000 time simulation

## Usage of the visualization tool
To convert the ID to table, you can use visualization tool.

```
java -jar ID2table.jar
```
or, double click ID2table.jar. 

![big_table](https://github.com/gumboshi/pzdr_saidai/blob/master/file/big_table.png)

1. change the table size and color

2. input the tableID

3. click "set" button 

![normal_table](https://github.com/gumboshi/pzdr_saidai/blob/master/file/normal_table.png)

You can save this image by "save" button to same folder as ID2table.jar.


## Licence

```
The MIT License (MIT)

Copyright (c) 2015 gumboshi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Author

[gumboshi](https://github.com/gumboshi)  [Movies](https://www.youtube.com/channel/UCzN57XZ4pEYWtEtOvNt00Ng)