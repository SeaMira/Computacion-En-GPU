# Curvas de nivel de un terreno

En la siguiente tarea se realizo la implementación de la visualización de un terreno en OpenGL. El terreno es una malla triangular generada aleatoriamente con una función fractal y cuenta con diversas características:

- Cámara de visualización en perspectiva y ortogonal
- Iluminación puntual y ambiental
- Respuesta al teclado y mouse.
- Interfaz de información e interacción del usuario

## Cámara de visualización en perspectiva y ortogonal

Inicialmente la cámara empieza en perspectiva en la mitad del mapa a una altura considerable. Para cambiar la perspectiva a ortogonal y viceversa se debe presionar el botón que indica aquello en la interfaz.

## Iluminación puntual y ambiental

Se tiene un ciclo de día y noche, mostrando incluso la hora que estaría representada. A falta de iluminación en la parte nocturna del ciclo, se implementó la capacidad de agregar una luz puntual en el lugar en que se encuentre la cámara (en perspectiva).

## Respuesta al teclado y mouse

Para mover la cámara se pueden utilizar las teclas WASD. Para ascender se puede utilizar la tecla SPACE y para descender la tecla LEFT SHIFT.

El mouse permitirá cambiar hacia donde está mirando la cámara. Para rotar el ángulo de visión se debe colocar el cursor en los bordes del viewport (pero dentro de él). Además se puede hacer zoom con la rueda del mouse.

#### IMPORTANTE: con la tecla R se regenera el terreno.

## Interfaz de información e interacción del usuario

Tal como se mencionó antes, esta interfaz despliega información de la escena:

- Posición de la cámara
- Hora del día ne la escena
- Un slider con la altura a la que se quiere hacer una curva de altura
- Opción de añadir una nueva curva o retirar la última
- Opción de agregar o quitar puntos de iluminación
- Cambio de cámara

### Compilación del proyecto

Abrir una terminal en la carpeta principal del proyecto (nombre T3).

Para crear una build y compilar se deben presionar los siguientes comandos:

```bash
cmake -S . -B ./build
cmake --build ./build -j 10
```

Luego para ejecutar se debe dar el siguiente formato de comando:

```bash
./build/surfaceLevelCurves.exe <grid size> <screen width> <screen height> <roughness>
```

Tal que:

- grid size: número entero que representa el tamaño del grid (grid cuadrado) en el que se genera el terreno, debe ser una potencia de 2 (129, 257, 513, 1025, etc). Con 257 ya se logran más de 10000 puntos.
- screen with: número entero que representa ancho de la ventana que corre el programa
- screen height: número entero que representa alto de la ventana que corre el programa
- roughness: float entre 0 y 1 que representa aspereza del terreno que se generará. 0.0 es un terreno sin picos

Ejemplo de uso:

```bash
./build/surfaceLevelCurves.exe 257 800 500 0.0
```

# Sobre los shaders

El proyecto tiene dos pipelines de con shaders diferentes: uno encargado del terreno y su iluminación y otro encargado de las curvas de nivel.

## Pipeline del terreno

Compila un vertex shader y un fragment shader. El primero entrega el color del vértice, la normal en esa posición, la posición de los puntos y la posición en coordenadas del mundo de la escena. El fragment shader se encarga de colocar las luces en escena y de asignar en cada pixel el color de acuerdo a la interpolación e iluminación.

## Pipeline de las curvas de nivel

Compila un vertex shader, un geometry shader y un fragment shader.

El primero simplemente entrega el color del vértice y su posición en coordenadas locales (no clipeadas ni en perspectiva).

El segundo se recibe las matrices de proyección, la cantidad de curvas y las alturas de las curvas, además de las alturas máximas y mínimas de las curvas. La primitiva que recibe es un triángulo y emite una línea. En base a esto revisa si el triángulo tiene vertices por arriba y por debajo de alguna de las curvas. En caso afirmativo, genera un segmento en el triángulo a la altura de las curva de un color que depende de la misma.

El fragment shader solo asigna el color generado en el geometry shader.
