##########################################################################################
# SYSTEM INFORMATION
##########################################################################################
import sys
print(sys.version)
print(sys.version_info)
print(sys.float_info)
##########################################################################################
# INTEGERS
##########################################################################################
a=20
b=10
print(a+b)
print(a-b)
print(a*b)
print(a/b)
print(a//b)
print(a%b)
print(a**b)
print(int(1.1))
print(int(True))
print(int(False))
print(bool(1))
print(bool(0))
print(True+False+1)
##########################################################################################
# REAL NUMBERS
##########################################################################################
pi=3.1415926536
radius=4.5
area=pi * (radius ** 2)
##########################################################################################
# COMPLEX NUMBERS
##########################################################################################
c=3.14 + 2.73j
c.real
c.imag
c.conjugate()
c * 2
c ** 2
d=1 + 1j
c - d
##########################################################################################
# FRACTIONS
##########################################################################################
from fractions import Fraction
f=Fraction(10,5)
# print(f.numerator,f.denominator,f)
##########################################################################################
# STRINGS
##########################################################################################
str1='hello world'
str2="hello world"
str3="""hello world"""
str4="""hello\nworld"""

greet_old='Hello %s!'
greet_old % 'Fabrizio'
greet_positional='Hello {} {}!'
greet_positional.format('Fabrizio','Romano')
greet_positional_idx='This is {0}! {1} loves {0}!'
greet_positional_idx.format('Python','Fabrizio')
greet_positional_idx.format('Coffee','Fab')
keyword='Hello,my name is {name} {last_name}'
keyword.format(name='Fabrizio',last_name='Romano')

from math import pi
# print(str1,str2,str3,str4,pi)
##########################################################################################
# TUPLES
##########################################################################################
t=()
type(t)
one_element_tuple=(42,)
three_elements_tuple=(1,3,5)
a,b,c=1,2,3
a,b,c
3 in three_elements_tuple
a,b=1,2
c=a
a=b
b=c
a,b
a,b=0,1
a,b=b,a
a,b
##########################################################################################
# LIST
##########################################################################################
[]
list()
[1,2,3]
[x + 5 for x in [2,3,4]]
list((1,3,5,7,9))
list('hello')

a=[1,2,1,3]
a.count(1)
a.extend([5,7])
a.index(1)
a.insert(0,17)
a.pop()
a.pop(3)
a.remove(17)
a.reverse()
a.sort()
a.clear()
a.append(100)
a.extend((1,2,3))
a.extend('...')
a=[1,2,1,3]
min(a)
max(a)
sum(a)
len(a)
b=[6,7,8]
a + b
a * 2

from operator import itemgetter
a=[(5,3),(1,3),(1,2),(2,-1),(4,9)]
sorted(a)
sorted(a,key=itemgetter(0))
sorted(a,key=itemgetter(0,1))
sorted(a,key=itemgetter(1))
sorted(a,key=itemgetter(1),reverse=True)
##########################################################################################
# BYTE ARRAY
##########################################################################################
bytearray() # empty bytearray object
bytearray(10) # zero-filled instance with given length
bytearray(range(5)) # bytearray from iterable of integers
name=bytearray(b'Lina') #A - bytearray from bytes
name.replace(b'L',b'l')
name.endswith(b'na')
name.upper()
name.count(b'L')
##########################################################################################
# SET TYPES
##########################################################################################
small_primes=set() # empty set
small_primes.add(2) # adding one element at a time
small_primes.add(3)
small_primes.add(5)
small_primes.add(1) # 1 is not a prime!
small_primes.remove(1) # so let's remove it
3 in small_primes # membership test
4 in small_primes
4 not in small_primes # negated membership test
small_primes.add(3) # trying to add 3 again
bigger_primes=set([5,7,11,13]) # faster creation
small_primes | bigger_primes # union operator `|`
small_primes & bigger_primes # intersection operator `&`
small_primes - bigger_primes # difference operator `-`
small_primes={2,3,5,5,3}
small_primes=frozenset([2,3,5,7])
bigger_primes=frozenset([5,7,11])
small_primes & bigger_primes # intersect,union,etc. allowed
##########################################################################################
# DICTIONARY
##########################################################################################
a=dict(A=1,Z=-1)
b={'A': 1,'Z': -1}
c=dict(zip(['A','Z'],[1,-1]))
d=dict([('A',1),('Z',-1)])
e=dict({'Z': -1,'A': 1})
a == b == c == d == e # are they all the same?
list(zip(['h','e','l','l','o'],[1,2,3,4,5]))
list(zip('hello',range(1,6))) # equivalent,more Pythonic
d={}
d['a']=1 # let's set a couple of (key,value) pairs
d['b']=2
len(d) # how many pairs?
d['a'] # what is the value of 'a'?
d # how does `d` look now?
del d['a'] # let's remove `a`
d['c']=3 # let's add 'c': 3
'c' in d # membership is checked against the keys
3 in d # not the values
'e' in d
d.clear() # let's clean everything from this dictionary
d=dict(zip('hello',range(5)))
d.keys()
d.values()
d.items()
3 in d.values()
('o',4) in d.items()
d.popitem() # removes a random item (useful in algorithms)
d.pop('l') # remove item with key `l`
d.pop('not-a-key','default-value') # with a default value?
d.update({'another': 'value'}) # we can update dict this way
d.update(a=13) # or this way (like a function call)
d.get('a') # same as d['a'] but if key is missing no KeyError
d.get('a',177) # default value used if key is missing
d.get('b',177) # like in this case
d.get('b') # key is not there,so None is returned
d={}
d.setdefault('a',1) # 'a' is missing,we get default value
d.setdefault('a',5) # let's try to override the value
d={}
d.setdefault('a',{}).setdefault('b',[]).append(1)
##########################################################################################
# IF STATEMENTS
##########################################################################################
c=5
if c<5:
    print("c<5")
elif c>5:
    print("c>5")
else:
    print("c=5")

print("c<5") if c<5 else print("c>5 or c=5")
##########################################################################################
# FOR LOOP
##########################################################################################
for i in range(5):
    print(i*5,end=" ")

for i in [1,2,3,4,5,]:
    print(i,end=" ")

surnames=['Rivest','Shamir','Adleman']
for i in range(len(surnames)):
    print(i,surnames[i],end=" ")

for i,z in enumerate(surnames):
    print(i,z,end=" ")

people=['Jonas','Julio','Mike','Mez']
ages=[25,30,31,39]
nationalities=['Belgium','Spain','England','Bangladesh']
for position in range(len(people)):
    person=people[position]
    age=ages[position]
    print(person,"\t",age)
for person,age,nationality in zip(people,ages,nationalities):
    print(person,age,nationality)
for data in zip(people,ages,nationalities):
    person,age,nationality=data
    print(person,age,nationality)
##########################################################################################
# WHILE LOOP
##########################################################################################
n=39
remainders=[]
while n > 0:
    remainder=n % 2 # remainder of division by 2
    remainders.append(remainder) # we keep track of remainders
    n //= 2 # we divide n by 2
    # reassign the list to its reversed copy and print it
remainders=remainders[::-1]
print(remainders)

while n > 0:
    n,remainder=divmod(n,2)
    remainders.append(remainder)
    # reassign the list to its reversed copy and print it
remainders=remainders[::-1]
print(remainders)

position=0
while position < len(people):
    person=people[position]
    age=ages[position]
    print(person,age)
    position += 1

items=[0,None,0.0,True,0,7] # True and 7 evaluate to True
found=False # this is called "flag"
for item in items:
    print('scanning item',item)
    if item:
        found=True # we update the flag
    break
if found: # we inspect the flag
    print('At least one item evaluates to True')
else:
    print('All items evaluate to False')

from datetime import date,timedelta
today=date.today()
tomorrow=today + timedelta(days=1) # today + 1 day is tomorrow
products=[
{'sku': '1','expiration_date': today,'price': 100.0},
{'sku': '2','expiration_date': tomorrow,'price': 50},
{'sku': '3','expiration_date': today,'price': 20},
]
for product in products:
    if product['expiration_date'] != today:
        continue
    product['price'] *= 0.8 # equivalent to applying 20% discount
    print('Price for sku',product['sku'],'is now',product['price'])

import numpy as np
from scipy import stats
a = np.arange(10)
stats.describe(a)
DescribeResult(nobs=10, minmax=(0, 9), mean=4.5,
               variance=9.166666666666666, skewness=0.0,
               kurtosis=-1.2242424242424244)
b = [[1, 2], [3, 4]]
stats.describe(b)
DescribeResult(nobs=2, minmax=(array([1, 2]), array([3, 4])),
               mean=array([2., 3.]), variance=array([2., 2.]),
               skewness=array([0., 0.]), kurtosis=array([-2., -2.]))



