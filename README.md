# Archeo

**Archeo** est une application pouvant être utilisée soit en **mode développement** avec Python, soit en **version desktop packagée** avec **PyInstaller** et **Electron**.

---

## Sommaire

- [Présentation](#présentation)
- [Prérequis](#prérequis)
- [Utilisation en mode développement](#utilisation-en-mode-développement)
- [Build de la version desktop](#build-de-la-version-desktop)
  - [1. Build du backend Python](#1-build-du-backend-python)
  - [2. Build du frontend Electron](#2-build-du-frontend-electron)
  - [3. Copie des fichiers internes](#3-copie-des-fichiers-internes)
  - [4. Lancement de l’application](#4-lancement-de-lapplication)
- [Structure utile du projet](#structure-utile-du-projet)
- [Remarques](#remarques)
- [Commandes rapides](#commandes-rapides)
- [Auteur](#auteur)

---

## Présentation

Le projet **Archeo** peut être utilisé de deux façons :

### 1. Mode développement
Le backend Python est lancé directement avec `app.py`.

### 2. Mode desktop
Le backend Python est transformé en exécutable avec **PyInstaller**, puis intégré dans une application **Electron** afin de produire une version bureau Windows.

---

## Prérequis

Avant de commencer, assurez-vous d’avoir installé :

- **Python**
- **pip**
- **Node.js**
- **npm**
- Les dépendances Python du projet
- Les dépendances du frontend Electron

---

## Utilisation en mode développement

Depuis la racine du projet, créez un environnement virtuel :

```bash
python -m venv venv
```

Activez ensuite l’environnement virtuel.

### Sous Windows
```bash
.\venv\Scripts\activate
```

Installez les dépendances Python :

```bash
pip install -r requirements.txt
```

Lancez ensuite Archeo :

```bash
python app.py
```

---

## Build de la version desktop

La version desktop repose sur deux parties :

- un **backend Python** packagé avec **PyInstaller**
- un **frontend desktop** packagé avec **Electron**

---

### 1. Build du backend Python

Ouvrez un premier terminal et placez-vous à la racine du projet.

Suivez ensuite les étapes indiquées dans le fichier :

```txt
backend_build.txt
```

Une fois le backend généré :

1. renommez l’exécutable obtenu en :

```txt
backend.exe
```

2. copiez ce fichier dans le dossier :

```txt
/electron/backend_dist
```

---

### 2. Build du frontend Electron

Ouvrez un second terminal et placez-vous dans le dossier :

```txt
/electron
```

Suivez ensuite les étapes indiquées dans le fichier :

```txt
command.txt
```

Cela permettra de générer la version desktop Electron.

---

### 3. Copie des fichiers internes

Une fois le build Electron terminé, rendez-vous dans le dossier :

```txt
/electron/dist/win_unpacked/resources/backend
```

Copiez ensuite dans ce dossier le répertoire :

```txt
/internal
```

qui se trouve dans :

```txt
/Archeo/dist
```

> Ce dossier est créé après le build du backend avec PyInstaller.

---

### 4. Lancement de l’application

Une fois toutes les étapes terminées, lancez :

```txt
Archeo.exe
```

Vous pourrez alors utiliser le logiciel.

---

## Structure utile du projet

Voici un exemple de la structure importante du projet :

```txt
Archeo/
│   app.py
│   backend_build.txt
│   engine.py
│   engine_meshlab.py
│   index.html
│   predictor_arcface.py
│   README.md
│   requirements.txt
│   segment.py
│
├── assets/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── electron/
│   │   command.txt
│   │   main.js
│   │   package.json
│   │   README.md
│   │
│   └── backend_dist/
│       └── backend/
│
├── _captures/
└── _uploads/
```

---

## Remarques

- Le backend doit être correctement généré avant le build Electron.
- L’exécutable du backend doit impérativement être renommé en **`backend.exe`**.
- Le dossier **`internal`** doit être copié manuellement après le build si nécessaire.
- En cas de problème de dépendances, vérifiez que :
  - le `venv` est bien activé ;
  - les dépendances Python sont bien installées ;
  - les dépendances npm du dossier `electron` sont bien installées.

---

## Commandes rapides

### Mode développement

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Build desktop

1. Suivez les étapes dans `backend_build.txt`
2. Renommez l’exécutable généré en `backend.exe`
3. Copiez `backend.exe` dans `/electron/backend_dist`
4. Suivez les étapes dans `electron/command.txt`
5. Copiez `/Archeo/dist/internal` dans `/electron/dist/win_unpacked/resources/backend`
6. Lancez `Archeo.exe`

---
## Liens
Lien de téléchargement du logiciel : https://drive.google.com/file/d/1p79jJQ0PcAqf08LT8ItTAb_gdUGnzt7k/view?usp=drive_link

Preview vidéo : https://drive.google.com/file/d/1GCA3sa65FVWNi_lpYvYGAK-Esamn19E4/view?usp=drive_link

Database d'images annotées : https://drive.google.com/file/d/1IoGinySxOuENC3Es-r4TPzL_Kdvx-vCO/view?usp=drive_link

Classification model (best_model_traced.pt) : https://drive.google.com/file/d/1TJkiDjuUBn9G6V7omAMNO1MfQHo6ic2B/view?usp=drive_link

Segmentation model (seg.pt) : https://drive.google.com/file/d/15aoyNV8pdBPY_p9RhcB7O7RoYCJFZ7JM/view?usp=sharing

---
## Auteur

Projet réalisé par **Loann KAIKA**.
