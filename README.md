# cosp

A Prolog-like logic inference system with cost-aware reasoning.

```
$ cat program.plc
[1] f(x?) :- g(x?).
[2] g(p*).
[4] f(p*).
$ cosp program.plc
?- f(x?).
x = p*
3.
?-
```
