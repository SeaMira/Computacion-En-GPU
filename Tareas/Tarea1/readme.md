# Tarea 1: Matrix en CPP

Esta es la librería pedida a implementar para la tarea 1 de Computación en GPU: Matrix.

## Requisitos

Antes de comenzar, asegúrate de tener instalado CMake. Este proyecto ha sido testeado con CMake versión 3.x o superior.

## Compilación del Proyecto

Para compilar el proyecto, sigue estos pasos desde la raíz del directorio del proyecto:

### Configuración de CMake

Para preparar el entorno para la compilación, ejecuta:

```bash
cmake -S . -B ./build
```

### Compilación

Para compilar el proyecto, utiliza el siguiente comando:

```bash
cmake --build ./build -j 10
```

El flag -j 10 permite la compilación paralela usando 10 hilos, lo que puede acelerar el proceso de compilación en máquinas multicore.

### Ejecución de Tests

Una vez compilado el proyecto, puedes ejecutar los tests con `ctest` utilizando:

```bash
ctest --test-dir ./build --output-on-failure
```

Este comando ejecuta todos los tests definidos y muestra la salida solo cuando un test falla, facilitando la depuración.

## Preguntas de la Tarea

### ¿Afectaría en algo el tipo de dato de su matriz?, ¿Qué pasa si realiza operaciones de multiplicación tipo de dato integer en vez de double?

Claro que el tipo de dato afecta a la matriz, en distintos aspectos. En características "físicas", por ejemplo, hay tipos de datos que ocupan más espacio que otros, char, int, long long, etc. Entonces podría ocurrir que matrices muy grandes con un tipo de dato que ocupe mayor espacio que un double vana a llenar el espacio disponible para variables de programa mucho más rápidamente.

A lo anterior hay que sumarle la otra característica que el cambio de dato implica: mayor o menor presición. Un long double por ejemplo ocupa mayor espacio, pero por lo mismo se tiene mayor precisión para valores de decimales.

Un entero (integer) tiene, al igual que el double, 32 bits de espacio, pero no se utiliza para valores decimales, es decir, no deben representar valores con precisión infinitesimal. Por esto entonces los cálculos son exactos, y la única problemática de la que hay que preocuparse es de los overflow y underflow, y no de los cálculos de punto flotante como en caso de double.

En caso de la multiplicación ocurre lo mencionado: en enteros no existirán problemas de precisión, sin embargo, en doubles (que buscan la precisión en decimales) pueden ocurrir errores, como problemas de redondeo, entregando resultados inexactos.

### Si se empezaran a usar números muy pequeños o muy grandes y principalmente números primos, ¿Qué ocurre en términos de precisión?

Si se ocuparan números muy pequeños podrían empezar a ocurrir problemas de redondeo en los cálculos (en la resta o multiplicación) llevando a aproximaciones a 0 o simplemente valores incorrectos.

Si son muy grandes puede ocurrir overflow o underflow, donde se superan los números máximos representables en bits según el tipo de datos.

Si principalmente son números primos se puede decir simplemente que se debe tener mucho cuidado con el manejo de ellos, pues al ser primos no se pueden representar como la multiplicación de factores de manera que se pueda reorganizar el espacio, además de que hay que fijarse en cómo realizar cálculos con ellos para mantener su condición de factor primo.

### ¿Pueden haber problemas de precisión si se comparan dos matrices idénticas pero con diferente tipo? (Matrix p1 == p2)

Sí pueden haber problemas de precisión simplemente por la forma en que se almacenan los números de punto flotante. Hay tipos de dato que pueden representar valores de maneras diferentes (si no están normalizados) o simplemente por la forma en que lo almacenó, y al compararse con la otra matriz que tiene otra representación se detectaría un error.
Lo mejor entonces es comparar números considerando un posible epsilon de diferencia

## Autor

Sebastian Andrés Mira Pacheco
