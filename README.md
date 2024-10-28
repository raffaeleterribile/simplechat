# Simple Chat

Una semplice chat per sperimentare con gli LLM.
Ho usato Microsoft Phi3 Mini come modello di intelligenza artificiale.
Ho creato 4 modi diversi di usarla:
1. lanciare il file chat.py
2. avviare un server REST con server.py
3. avviare un server realtime ed interrogarlo tramite console
4. avviare un server realtime ed avviarlo tramite client realtime

## Metodo 1
Il metodo 1 utilizza un'interfaccia realizzata con Gradio

## Metodo 2
Il metodo 2 realizza un server con FastAPI definendo un unico endpoint acceduto da un comando curl

## Metodo 3
Il metodo 3 realizza un server realtime realizzato con gRPC ed una applicazione console

## Metodo 4
Il metodo 4 realizza un server realtime realizzato con gRPC ed una applicazione realizzata con Gradio
