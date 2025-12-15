# proyecto-fatiga

Este proyecto implementa un Pipeline de Visión Computacional de Baja Latencia.

En lugar de usar redes neuronales pesadas que requieren GPU, opté por Clasificadores en Cascada (Haar Cascades) optimizados para CPU. Esto permite que el sistema corra en navegadores web y dispositivos móviles sin lag.

El algoritmo no 'aprende' en vivo, sino que analiza la geometria del ojo en tiempo real: detecta la presencia de patrones oculares y calcula una Métrica de Fatiga Temporal. Si la consistencia de detección cae por debajo del umbral calibrado (min_neighbors), se dispara la alerta. 

El unico detalle es que fallé en implementar una alarma sonora, pues la reproduccion del sonido en google chrome es bastante complicada para terceros.

¿En que parte se implementa el ML?

El modelo tiene que saber reconocer una cara primeramente para si quiera analizar la geometria ocular, es decir, reconocer cejas, nariz boca, en miles de datos de entrada, este trabajo lo hace google, y yo, tomé su modelo ya entrenado y le añadi una funcion de utilidad.
