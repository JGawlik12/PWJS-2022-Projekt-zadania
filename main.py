import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from MyKMeans import MyKMeans

dataset = pd.read_csv("Mall_Customers.csv") # wczytanie danych
X = dataset.iloc[:, [3, 4]].values # pobranie 3,4 kolumny
klasy = [ilosc for ilosc in range(2, 8)] #stworzenie tablicy z wartosciami 2-8


#iterujemy tablice klasy
for liczba_klas in klasy:
    fig = plt.figure() #tworzenie nowego okna na wykres
    ax = fig.add_subplot() #dodanie nowego subplota do okienka
    ax.set_xlim([-0.3, 1]) #ustawienie zakresu wartosci osi x
    ax.set_ylim([0, len(X) + (liczba_klas + 1) * 10]) #ustawienie zakresu wartosci osi y (liczna probek + nieco pikseli na odstep)


    #kmeans = KMeans(n_clusters=liczba_klas, random_state=10)
    #etykiety = kmeans.fit_predict(X)

    mykmeans = MyKMeans(n_clusters=liczba_klas)
    etykiety = mykmeans.fit_predict(X)  # stworzenie klastrow


    silhouette = silhouette_score(X, etykiety) #obliczenie wspolczynnika shoulette
    print("Dla ", liczba_klas, " klas sredni wspolczynnik wynosi :", silhouette)
    silhouette_wartosci = silhouette_samples(X, etykiety) #obliczenie wspolczynnika shoulette dla kazdego punktu
    y_dolne = 10 #odstep na dole wykresu

    #rysowanie wykresu
    for i in range(liczba_klas):
        klaster = silhouette_wartosci[etykiety == i] #pobranie wartosci z danego klastra
        klaster.sort() #sortowanie
        rozmiar_klastra = klaster.shape[0] #pobranie liczby punktow
        y_gorne = y_dolne + rozmiar_klastra #wysokosc nowego klastra
        kolor = cm.nipy_spectral(float(i) / liczba_klas) #stworzenie kolorow
        ax.fill_betweenx(np.arange(y_dolne, y_gorne), 0, klaster, facecolor=kolor, edgecolor=kolor, alpha=0.7) #rysowanie klastra
        ax.text(-0.05, y_dolne + 0.5 * rozmiar_klastra, str(i)) #rysowanie indeksu
        y_dolne = y_gorne + 10 #przesuniecie w gory z odstepem 10

    ax.set_title("Wykres Silhouette.") #ustawienie tytulu wykresu
    ax.set_xlabel("Wartosci wspolczynnika Silhouette.") #ustwaienie etykiety osi x
    ax.set_ylabel("Etykiety.")  #ustwaienie etykiety osi y
    ax.axvline(x=silhouette, color="red", linestyle="--")  #rysowanie linii wyznaczajacej wspolczynnik shoulette
    ax.set_yticks([]) #usuniecie wartosci z osi y
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1]) #ustawienie wartosci osi x

plt.show() #pokazanie wykresow