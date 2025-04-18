!--
  This file is part of code_saturne, a general-purpose CFD tool.

  Copyright (C) 1998-2024 EDF S.A.

  This program is free software; you can redistribute it and/or modify it under
  the terms of the GNU General Public License as published by the Free Software
  Foundation; either version 2 of the License, or (at your option) any later
  version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
  details.

  You should have received a copy of the GNU General Public License along with
  this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
  Street, Fifth Floor, Boston, MA 02110-1301, USA.
-->

\page cs_dg_further_reading Further reading

[TOC]

Build and Tool chain
====================

Different parts of the code_saturne tool chain are built with different *programming languages*
and *libraries*.

- *Build system* based on GNU autotools: [autoconf](https://www.gnu.org/software/autoconf),
  [automake](https://www.gnu.org/software/automake/),
  - Requires optional *sh* shell and *m4* macro languages, some *make* file syntax, some
    *Python* code.
- *GUI* and scripts (*TUI*): [Python](https://www.python.org/),
  [PyQt](https://riverbankcomputing.com/software/pyqt/intro).
- *Preprocessor*: *C* compiler (C11 or above)
  - Optional mesh format libraries: *MED*, *CGNS*, *libCCMIO* (see install guide for details).
- *Solver*: *C* and *Fortran* compilers (C11 or above, Fortran 2008 or above.
  - Multiple optional external libraries: *MPI*, *PT-SCOTCH*, *MED*, *CGNS*, *Catalyst*,
    *CoolProp*, *PETSc*, and others (see install guide for details)

Recommended learning material
=============================

Depending on the part of the code one is working on, further reading and training
may be useful.

For the [Git](https://git-scm.com/) source code management system:
- *pro Git*: https://git-scm.com/book/en/v2

C programming
- Interactive tutorial https://www.learn-c.org/
- MIT OpenCourseWare (OCM): [Practical Programming in C](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-087-practical-programming-in-c-january-iap-2010/index.htm)
- Shorter course: [Scientific Programming in C](http://www.courses.physics.helsinki.fi/fys/cprog/)
  (see Lectures)
- For French readers, the IDRIS course is recommended: http://www.idris.fr/formations/c/
  - *MPI* and *OpenMP* Courses may also be found on the main course material page:
    [MPI_OpenMP](http://www.idris.fr/formations/supports_de_cours.html)

C++ programming
- Complete book on [Modern C++ programming](https://github.com/federico-busato/Modern-CPP-Programming) and associated subjects.
  This book touches on many interesting subjects, and is quite easy to read despite its large size.
- Another [Modern C++ for Absolute Beginners](https://github.com/jonalexjm/Books-C-and-C-plus/blob/main/Modern%20C%2B%2B%20For%20Absolute%20Beginners.pdf) book.

C++ parallel programming
- For parallel C++ programming allowing offload to accelerator devices, an
  [NVIDIA developer blog post](https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/)
  illustrates how this can be sone, with minimal impact to to code when switching
  from CPU to GPU, and a low dependency on vendor specific syntax such as that
  used in CUDA.
- Note that this requires C++17, and as more and more systems include compilers
  supporting C++20, newer constructs may be an option in the near future.

For visualization of code execution (Python, C, C++, and more), the following educational tool is
very nice: https://pythontutor.com.

C and C++ basics
================

A few C/C++ syntax elements are provided here. This list is not exhaustive,
but contains elements most often used in code_saturne.

<table>
<tr><th> <th> C
<tr><td>
Basic types
<td>
```{.cpp}
char
bool
int
float
double
```
<tr><td>
Basic math functions
<td>
```{.c}
(a)cos  (a)sin  (a)tan
cosh  sinh  tanh
exp  log
sqrt  pow(x, y)
abs  fabs
a%b
```
<tr><td>
Logical expressions
<td>
```{.c}
&&  ||  !
<  >  <=  >=  ==  !=
==  !=
```
<tr><td>
Function call
<td>
```{.c}
x = f(y);
g(a);
```
<tr><td>
Conditional
<td>
```{.c}
if (expr) {
  operations;
}
```
<tr><td>
Simple loops
<td>
```{.c}
for (i = 0; i < n; i++) {
  y /= 2.;
  z = x + i + y*5.;
}
```
<tr><td>
Complex loops
<td>
```{.c}
while (expr) {
  operations
}
```
<tr><td>
Loop control
<td>
```{.c}
for (i = 0; i < n; i++) {
  if (ignore_condition)
    continue;
  else if (exit_loop_condition)
    break;
}
```
<tr><td>
Variable declaration and initialization
<td>
```{.c}
int i, j,
    k = 0, l = 1;
double a = 1.;
```
<tr><td>
Array declaration and initialization
<td>
```{.c}
int tab[2][3] = {{1, 2, 3},
                 {4, 5, 6}};
printf("%d", tab[1][0]); // 4
```
</table>

It is important to keep in mind that C passes arguments by copy (so changing argument values in C has not effect unless that argument is a pointer or array
(see following sections), while in Fortran, variables are passed by reference, allowing their modification directly.

C and C++ languages
===================

C variable declarations
-----------------------

A variable may be initialized upon its declaration:

```{.c}
double a[4] = {1, 2, 3, 4};
double b[] = {1, 2, 3, 4};
double matrix[3][4] = {{1., 0., 0., 0.},
                       {0., 1., 0., 0.},
                       {0., 0., 1., 0.}};
static int call_id = 0;
const double pi = 4.*atan(1.), e = exp(1);
```

C Types and Structures
----------------------

C allows defining additional types, as well as structures.

- `typedef double cs_real_t` defines a `cs_real_t` type
  identical to `double`
- `typedef cs_real_t cs_real_3_t[3]` defines a `cs_real_3_t` type
  identical to an array of 3 `cs_real_t` types
  - indirectly equivalent to an array of 3 `double` types
- code_saturne makes use of this to define additional types (see especially
  \ref cs_defs.h and [integer type descriptions](@ref sec_prg_lang_integer_types)
- the `_t` postfix is a convention, which is recognized
  by some text editors (such as Emacs) for syntax coloring.

C++ adds the `using` syntax. For example, with a _stride_ template parameter:
- `using grad_t = cs_real_t[stride][3];`

Pointers and arrays
-------------------

Understanding pointers is essential in C, and still very useful in C++.

- In any language, variables are stored in memory.
- C allows access not only to a variable's value, but to its memory location
  (based on its memory model; this is usually a logical, not physical address)
  - A *pointer* is a variable referencing another variable's memory location.
  - In C, some operations cannot be done without handling pointers.
- For a given type, prefixing `*` to the variable's declaration
  declares it as a pointer to a variable of that type
- For a pointer, prefixing `*` to the pointer's name *dereferences* a pointer
  (that is accesses the value pointed to)
- Prefixing `&` to a variable obtains a reference (pointer) to that variable.

A simple example is best:
```{.c}
double x;
double a = 1.0, b = 2.0; // variables
double *p = NULL;

p = &a;                  // p points to a
x = *p;                  // x now has a's value
p = &b;                  // p points to b
*p = 12;                 // b's value is now 12.
```

Note that pointers in other languages (especially Fortran) are often more
restrictive, or include additional metadata.
- Pointers in C are simply a memory address.
- Associated type information (used for pointer arithmetic and checks) is defined
  in the source code and managed by the compiler, _not stored in the pointer_
 (i.e. no runtime introspection)

### C Pointers and arrays

- Array syntax may be used with pointers.
- Pointer "arithmetic" is based on the pointer's type (unless `void`).
- A single (but important) difference between pointers and arrays:
  - An array declaration implies local memory allocation, a pointer does not.
  - This is reflected by the different behavior of the `sizeof`
- In addition to the following example, check the tutorial at:
  https://boredzo.org/pointers/

```{.c}
double x;
double a[3] = {1.0, 2.0, 3.0};
double *p = a;            // p points to beginning of a

x = *p;                   // same as a[0]
x = p[0];                 // same as a[0]
x = p[2];                 // same as a[2]
p[1] *= 2;                // a[1] *= 2
p++;                      // point to next value
x = *p;                   // same as a[1]
```

### Remarks on pointers

Character strings do not have a dedicated type in C:
- Strings are simply arrays or pointers to characters.
- Strings end when a character with code 0 (`\0`) is encountered

Pointers may be _cast_ from one type to another. For example:
```{.c}
int a[3][3] = {{11, 12, 13},
               {21, 22, 23},
               {31, 32, 33}};
int *p = (int *)a;
const int *q = (const int *)a;
```

C structures
------------

We will refer to other sources to detail usage of structures in code_saturne,
focusing here on specific aspects.

- Some structures are defined directly in a header (`.h`) file
  ```{.c}
typedef struct {
  int       n;           /* number of elements */
  double   *val;         /* list of element values */
} cs_simple_struct_t;
  ```

  - Such structures may be used normally, with full access to members from any
    file including the header

- Other structures are defined in in a source (`.c`) file
  ```{.c}
typedef struct _cs_opaque_t {
  int       n;           /* number of elements */
  double   *val;         /* list of element values */
};
  ```
  with a matching entry in a header (`.h`) file
  ```{.c}
typedef struct _cs_opaque_t cs_opaque_t;
  ```
  - Such structures may be used normally inside the source file, but their members
    are _hidden_ from within other files.
  - This allows protecting structure members, ensuring they are only accessed from
    within the associated module
  - If the file is large and needs to be split, we place the definition in a
    separate header file, accessed only from a restricted set of files
    (see for example `src/base/cs_matrix*.h`).

Using opaque structures has advantages:
- It conveys the information to most users that they can use the structure without
  worrying about its internals
- Structure internals can be modified without breaking compatibility (changing
  access functions)

It also has some disadvantages:
- Access is more cumbersome, requiring functions.
- Due to function call overheads, many calls to simple functions in a loop are
  more costly than  direct access, or than a function which loops internally

C++ classes
-----------

C++ classes are an extension of C structures, providing:
- Functions associated to a structure (_methods_, or _member functions_).
- Automatic initialization and destruction of a structure and its members.
- Finer control on structure member access (`public` and `private` qualifiers),
  which are an alternative to using either fully opaque of fully accessible
  C structures.
- A class may be _inherited_ from another, allowing the extension of adaptation
  of the base class while automatically reusing all unspecified (i.e. _overriden_)
  elements of that base class.

In C++, a `class` whoses members are all public and a `struct` are identical
(so a C++ `struct` is an extension of a C `struct`, as it can contain
specific constructors, destructors, and member functions.

```{.cpp}
class cs_simple_class {
public:
  int       n;           /* number of elements */
  double   *val;         /* list of element values */

public:
  // Constructor
  cs_simple_class(int  n_elts) : n(n_elts) : {
    CS_MALLOC(val, n, double);
  }

  // Destructor
  ~cs_simple_class() {
    CS_FREE(val);
  }

  // Dot product
  double
  dot() {
    return cdot(n, val, val);
  }
};
```

Constructors have the same name as a class, and no return type. Multiple
constructors with different arguments may be defined, when multiple
construction methods are desired.
Destructors have the same name as a class, preceded by `~`.

Other member functions can have any name, as a regular function.

A given instance of a class is called an _object_. For example
```{.cpp}
cs_simple_class a(n);
```
instanciates an object `a`, automatically calling its constructor.

Member functions can be called as follows:
```{.cpp}
double r = a.dot();
```

C storage class specifiers
--------------------------

A variable declaration can be combined with a \emph{storage class specifier}.

- `static` indicates the variable is "permanent":
   - Its values are saved from one call to the next.
- `extern` indicates we reference a variable, but its memory location is not
   defined here; for example:
   - `int option = 2;` in a _single_ (owning) `.c` file.
   - `extern int option;` in a `.h` or other `.c` file.
   This ensures the variable is accessible from multiple files, but defined in
   a single location.
   - For global variables, avoids conflicts (bugs) due to multiple copies
- `auto` is the default, so no point in specifying it.
- `register` recommends to keep an often-used variable in a _register_
   - Very fast, very small memory; the compiler does what it can / wants
- `volatile` indicates the variable may be modified by concurrent thread
  -  Reload from memory, not from cache.

### C const attribute

A variable declaration may specify one or more `const` attributes.

- A  `const` attribute indicates the function may not modify this variable
  - For a pointer, be careful where `const` is placed:
  ```{.c}
const double a[];       /* cannot modify
                           content of a */
const double *b;        /* same with b */
double *const c;        /* can not modify pointer
                           to c, but can modify
                           content pointed to */
const double *const c;  /* can modify neither pointer
                           nor content */
```

### C const attribute

- As variables are passed by copy, some constructs are not very useful.
- For example, `inline int f(const int *const t)` and
  `inline int f(const int *t)` are equivalent from the caller's side.
  - Only in the function body, the first syntax indicates we cannot locally modify
    the pointer to `t`.
  - Compilers allow mixing the first syntax in the body with one or the other in
    the prototype.
    - Proof if needed that for the caller, it is all the same.
  - From a readability standpoint, we prefer the second syntax, as it is
    less cluttered.
    - There may still be relics of the first syntax in code_saturne, especially
      in `src/gui`; choose more recent code examples, such as \ref cs_field.c;
  - It is strongly recommended to use `const` as much as possible
  - It can allow detecting unintentional variable modifications at compile time.

### C restrict attribute

A variable declaration may also be qualified with a `restrict` attribute.

 - The `restrict` attribute indicates that no other pointer refers to the same content.
   - It is part of the C standard but only an extension in C++.
     - The `[restrict]` syntax allowed for arrays in C is not supported by the known C++ `__restrict__` extensions, so should be avoided.
 - For example, with `int f(double *restrict a, double *restrict b)`, we tell
   the compiler that arrays a and b do not overlap.
   - Allows a better optimization, possibly vectorisation.
   - Most performance differences between C and Fortran are due to aliasing
    (forbidden in Fortran, assumed by default in C).
   - C _strict aliasing_ rules: between different types
     `double` and `int` for example, aliasing is _automatically_ forbidden.
 - This is useful only to help optimization of costly loops
   - If we forget to use this, we may lose some performance
   - If we incorrectly declare a variable `restrict`, we may have
     _undefined_ behavior...

### C functions

Like most programming languages, C allows grouping statements in functions.

- A function definition is composed of a _header_ and a _body_.
  - The _header_ describes the return type, the function name, and
    function arguments. if no value is returned or the function has no arguments,
    `void` is used to indicate this.
  - A function body contains the actual instructions of the function
- The following example function returns the dot product of 2 arrays
  ```{.c}
double f(int n, double x[], double y[])
{
  int i;
  double r = 0;
  for (i = 0; i < n; i++)
    r = r + x[i]*y[i];
  return r;
}
  ```

- Modern C strongly recommends functions be described by a _prototype_
  (i.e. interface), declared before defining or calling functions.
  - C++ requires this absolutely.
  - A function prototype resembles its header, ended by a semicolon (`;`).
- For the previous example, the matching prototype is:
  ```{.c}
double f(int n, double x[], double y[]);
  ```
- Only parameter types are required to match in the definition  and prototype,
  so compilers will not complain if names are different
  - But the code maintainers _will_ !
  - And the documentation generator will emit warnings

Prototypes are usually grouped in `header` files, inserted locally using the
`#include` preprocessor directive.

- If the `static` qualifier is prefixed to a function
  declaration, that function is defined locally only
  - In this case, prototypes are not necessary (functions referenced by others
     must appear first).
  - Functions with the same name may be used in different files with no risk.
- Using `static inline`, the function body is copied at each call
  - Avoids call overhead for short functions, leads to larger code.
  - `inline` without `static` is tricky: see a more complete C course, or avoid it.
- In code_saturne, some simple computation functions are defined as
  `static inline`;
  - Their definition appear in header (`.h`) files in this case
  - See for example \ref cs_math.h.

- In code_saturne, many low-level functions are defined as `static`
  - When they are only needed locally, this avoids cluttering the
    Doxygen documentation with low-level functions
  - This also allows using shorter names, without worrying about
    `cs_<module>_` "namespace" issues.
    - As per static (i.e. file-local) global variables, we simply prefix the names
      of those functions with an underscore: `_`
  - As those functions do not require prototypes, they are defined at the beginning
    of the file; if function `b` calls function `a`, then `a` must be
    defined first.
  - If such functions may become useful elsewhere, it is best to make
    them non-static (i.e. global), move them in the file, and add a prototype
    rather than to adopt a copy-paste programming style...

- A function is called by providing its name and arguments
  ```{.c}
r = f(3, x, y); // returns r
g(x);           // returns no value
s = h();        // takes no argument
  ```

- In C, functions <span style="color:red">pass arguments by value</span>.
  - Item array contents, or values referenced by pointers may be modified normally
  - Non-pointer (or array) arguments are copied, so the original is
    unchanged in the calling code

Example of call by value semantics:
```{.c}
/* callee function */
void f(double x, double y[2]) {
  x = x/2;
  y[1] = x;
}

/* caller function */
void g(void) {
  double x = 10, y[] = {1., 2.};  /* initialization */

  f(x, y);                        /* call to f */

  /* now x = 10, y[] = {1., 5.} */

  ...
}
```
- In C++, functions can pass arguments either by value or by reference.
  - By default, the behavior is the same as in C (pass by value).
  - Using `&` before an argument's name, that argument will be passed by
    reference.

Example of call by mixed semantics:
```{.cpp}
/* callee function */
void f(double x,              // by value
       double &y) {           // by reference
  x = x/2;
  y = x;
}

/* caller function */
void g(void) {
  double x = 10, y = 1.;  // initialization

  f(x, y);                // call to f

  // now x = 10, y = 5.

  ...
}
```

### C function pointers

Generic functions may be called using _function pointers_

- Don't be intimidated by the name
  - Some users don't even realize they are using function pointers.
- In practical terms, function pointers allow passing a function as an argument
  to a function
- To illustrate this, let us look at the examples in \ref cs_post.h and in
  \ref cs_user_postprocess.c.
  - ... not so hard, is it now ?

Memory management
-----------------

Explicit memory allocation operations return pointers

- Using the `malloc`, `realloc`, and `free` functions or similar
  - In code_saturne, the \ref CS_MALLOC, \ref CS_REALLOC, and \ref CS_FREE
    functions add type and result checking and instrumentation.
- Explicit allocation as described above is usually done on a memory area called the
  [heap](https://en.wikipedia.org/wiki/Memory_management#HEAP), which is a large,
  usually extensible memory area.
  - If memory is allocated but never freed, memory "leaks" then possibly runs out
  - Use [Valgrind](https://www.valgrind.org), the GCC and Clang
    [AddressSanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer)
    tools, or `export CS_MEM_LOG=mem.log` in your environment to check for this.
    As a special case, with `export CS_MEM_LOG=performance.log`, only statistics,
    including the number of non-freed pointers (if present) will be added to the usual
    `performance.log` file.
- Automatic allocation of variables and fixed-size arrays is done on a smaller
  memory area called the [stack](https://en.wikipedia.org/wiki/Stack_(abstract_data_type))
  - Does not incur any overhead (fast).
  - Automatically freed when variable goes out of scope.
  - Overwrites on the stack may crash even your debugger...
    - they also may crash _Valgrind_, but can be detected with the
      _AddressSanitizer_ tools.

C Preprocessing
---------------

Before the C compilation proper, a first stage replaces character sequences based
on several rules.

 - It is called the _preprocessor_
 - Directives start with `#`
   - `#include`, `#if`, `#ifdef`, `#ifndef`, `#define`,
- Allows defining _macros_
  - Using a common coding convention, we write them in capitals.
  - No need for `;` at the end of the line (or statement).
- Avoid macros for math operations, as they may have side effects
  (such as calling a function multiple times), and make debugging
  more difficult.

- Some macros are predefined; to know them, the solution is compiler
  dependent. With *gcc*, the following command is
  useful: `gcc -dM -E - < /dev/null`

- One of the main uses of the preprocessor is conditional compilation
  ```{.c}
#if defined (HAVE_MPI)
...
#endif
  ```

- To disable code containing comments, nothing beats:
  ```{.c}
#if 0
...
#endif
  ```

  - This avoids comment nesting issues, and some editors such as _vim_
    even colorize the block as a comment.

### C Preprocessor macros in code_saturne

code_saturne defines several preprocessor macros, among which the following:
  - \ref CS_F_(fname): access to field structure of field with
    canonical name `name`.

Two macros of special importance are:
  - \ref BEGIN_C_DECLS
    - In C++, expands to
      ```extern "C" {```
    - Empty in C.
  - \ref END_C_DECLS
    - In C++, expands to
      ```}```
    - Empty in C.

Using `extern "C"` in C++ tells the C++ compiler to generate C-linkable code,
with no [name mangling](https://en.wikipedia.org/wiki/Name_mangling).
Code enclosed in these sections can be called from C code.
If a header file includes C++ contructs with no C equivalents, that
code must also be protected by an
```{.cpp}
#ifdef __cplusplus
...
#endif
```
sequence so as to be ignored by the C compiler.
Starting with code_saturne 9.1, we will not try to ensure C compatibility
anymore, so these constructs are important mainly in regard to versions
9.0 and older.

### Preprocessors in various programming languages

Preprocessors do not exist in all "modern" languages, are often decried by purists,
but are very useful in C and C++ for optional library support.
- in Python, not missed, as we can use `try...import` sequences

C variable and function scoping
-------------------------------

Variables may have a local scope:

```{.cpp}
int f(int n, double x[]) {
  int i;
  i = 6;
  {
    int i, j; // i masks previous definition
    for (i = 0, j = 1; i < n; i++, j+= rand())
      x[i] += j;
  }
  /* i = 6 again */
  {
    int j; // the previous definition is unknown here !
    for (j = 0; j < n; j++)
      x[j] += 1.;
  }
}
```

### Advantages and precautions:

- Avoid multiple definitions of a variable on different levels
  - Check for compiler warnings: *definition shadows previous definition*
- Local definitions may improve readability.
- Local definitions ensure variables are _local_ (and thus automatically private)
  in OpenMP sections.

Since the C99 standard, variables may be defined in a
function body, or even in a control structure:

```{.cpp}
int f(int n, double x[]) {
  int i;
  i = 6;
  for (int j = 0; j < n; j++)
    x[j] += 1.;
  for (int j = 0; j < n; j++)
    x[j] += 1.;
  printf("value of j: %d\n", j); /* error, j not
                                    defined here */
}
```

The C scoping rules also allow definition of global variables.

- Declaring a variable in a source file outside a function makes it _global_
  - It is recommended to add an initialization to the declaration when
    possible, which is safer and simple than requiring additional initialization
    functions
- Adding the `static` qualifier makes that variable accessible only from the
  file in which it is declared.
  - Another similar variable in another file would be completely independent.
- If global visibility is desired, the definition should be unique,
  and the variable defined using the `extern` qualifier.
- An `extern const` qualifier may be used to make the variable read only.
  - Useful for pointers to structures, allowing safe reading of structure
    members, but modification only though a specific function (see
    the handling of \ref cs_glob_time_step in `src/base/cs_glob_time_step.c`
    `src/base/cs_glob_time_step.h` for example).

### Main global variables in code_saturne

- In code_saturne, we try to minimize the use of global variables, but a few
  are used, as placing them in a structure would be cumbersome for C/Fortran
  interoperability.

```{.c}
(int) cs_glob_n_ranks                            // cs_defs.h
(int) cs_glob_rank_id                            // cs_defs.h

(cs_matrix_t} cs_glob_matrix_default             // cs_matrix.h
(cs_matrix_structure_t} cs_glob_matrix_default_struct

(cs_domain_t) cs_glob_domain                     // cs_domain.h
(cs_mesh_t) cs_glob_mesh                         // cs_mesh.h
(cs_mesh_quantities_t) cs_glob_mesh_quantities   // ...

(cs_time_step_t) cs_glob_time_step               // cs_time_step.h

(cs_field_pointer_val_t) cs_glob_field_pointers  // ...
```

C Undefined behavior
--------------------

Some ambiguous constructions lead to what is called _undefined behavior_.

- The compiler is free to do whatever it wants in such cases.
  - different compilers may exhibit different behaviors
  - <span style="color:red">Avoid at all costs</span>, but in general,
    there is nothing to worry about as long as the code's conventions
    are followed.

- Example: incorrect character string usage
  ```{.c}
char *p = "code_saturne"; // forbidden in C++.
p[0] = 'C'; // undefined behavior due to above
            // (but works with most compilers)
  ```
- Correct character string usage
  ```{.c}
char p[] = "code_saturne"; // array, not just pointer
p[0] = 'C'; // OK
  ```

- Example: division by zero
  ```{.c}
int x = 1;
return x/0; // undefined behavior
  ```

- Example: out of bounds array access
  - this detected by _Address Sanitizer_, but not by _Valgrind_:
     as `arr` is declared as a local array, it is instanciated on the _stack_,
     not the _heap_.
  ```{.c}
int arr[4] = {0, 1, 2, 3};
int j = arr[5]; // stack buffer overflow
  ```

- Example: out of scope return value
  ```{.c}
  *double badarray(void) {
    double t[] = {0, 1, 2};
    return t;  // memory location freed on return
  }
  ```

- Example: undefined (or not always defined) return value
  - Very easy to avoid, as current compilers emit a warning.
  - You check compiler warnings, of course ?
  ```{.c}
int
f(int x)
{
  if (x < 1) {
    return -x;
  }
}  /* undefined behavior if x >= 1 */
  ```

- Example: incrementation before/after use (note 84) C11 standard?
  ```{.c}
printf("%d %d\n",
       ++n, pow(2, n));  /* is n incremented
                            before or after
                            calling power ? */
i = ++i + 1;
a[i++] = i; /* is i incremented before or
               after assignment ?*/
/* The constructs below are safe: */
i = i + 1;
a[i] = i;
a[i++] = j;
  ```

- _Rule of thumb_: to be safe, avoid incrementation operators on an index
   which appears multiple times in an expression.

C Pragmas
---------

Another type of element may start with a `#`, but is not related to the
preprocessor: `pragmas`
- `#pragma omp ...` for optional thread/task parallelism using the _OpenMP_ model
- `#pragma disjoint(<variable list)` for directives specific to optimizations
   using the IBM XL compilers (at least in older versions); in most cases,
   the `restrict` keyword is a more portable alternative
- `#pragma GCC ...` for directives specific to GCC

The most frequent pragmas in code_saturne are related to _OpenMP_ parallelism
- They are used only if this programming model is activated

In a general manner, a `pragma` not known to a given compiler is ignored

C Decorators
------------

In a few rare places, we use _decorators_

- In the following example, the `__attribute__` decorator is used to specify
  that the function behaves like `printf` regarding its arguments, so as to
  benefit from compiler argument checking
  ```{.c}
#if defined(__GNUC__)
int
bft_printf(const char  *const format,
           ...)
  __attribute__((format(printf, 1, 2)));
#else
int
bft_printf(const char  *const format,
           ...);
#endif
```

Various decorators exist, but we do not use them much in code_saturne,
as they are _not portable_.

C++ attributes
--------------

C++ allows a more portable alternative to C decorators, called
[attributes](https://en.cppreference.com/w/cpp/language/attributes).

Attributes most often used in code_saturne include `[[fallthrough]]`, which
indicates that if a `swich`/`case` entry continues to the next case without a
`break` statement, this is intentional, and the compiler should not emit a
warning, and `[[maybe_unused]]`, which also avoids a comppiler warning if a
function parameter is not used in a given case).
