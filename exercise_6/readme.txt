Solution by: Falco Lentzsch(685454),Konrad von Kuegelgen(676609), Jesse Kruse(675710)

Bonus:
1.  Als erster haben wir Bonusaufgabe_2 als Verbesserung versucht. Dazu haben wir einen Faktor Lamda während des Trainingseingeführt.
    Dieser wird mit dem Offset multipliziert um mehr Zufall in die Bestimmung der neuen Keypoint Positionen zu bringen.
    Dabei haben wir bei einem zufälligen Lamda im Bereich von 0,8 - 1,2, das beste Ergebnis erhalten. Siehe optimization_1.pdf
    Wählt man diesen Bereich zu groß, so wurde das Netzt durch große Ausreißer eher verwirrt während des Lernprozesses.
    Auf Basis dieses Ergebnisses haben wir die folgende Optimierung vorgenommen

2.  Als nächstes haben wir Bonusaufgabe_3 umgesetzte und bei der Evalution versucht anstatt auf 9 Bilder zu Testen,
    auf 10*9 Bildern zu testen und dafür die Koordinaten aus den Trainingsbildern ebenfalls um einen Faktor Lamda zu verschieben.
    Das Ergebnis kann optimization_2.pdf entnommen werden, wobei die letzten beiden Seiten den Vergleich darstellen.

3.  Nun versuchten wir durch Variationen des Netzes bessere Ergebnisse zu erzielen. Eine Verkleinerung/Vergrößerung (Encoder/Decoder Stil)
    brachte keine Verbesserung. Eine Vergrößerung der Feature Maps auf 128 hatte ebenfalls einen Nachteil. Ähnliche Effekte der Channel-
    Veränderung erhielten wir auch bei unterschiedlichen Kernelgrößen (auch alternierend, Info später) und Netzschichten. Unsere neue Architektur 
    behält also weiterhin die 64 in/ou-channel bei. Die vielversprechenste Netzvariante erhielten wir durch das Löschen des Dritten Layers, 
    sowie einer Erhöhung der Kernelgröße bei der mittleren Faltung (3,5,3,3). Ein größerer Faltungs-Kernel vergrößert das rezeptive Feld der
    Feature Map, welche durch die fehlende Tiefe (gelöschter Layer) bereits für wichitge Features kodieren sollte. In der Auswertung existieren
    auch am Ende Sprünge im Loss, aber der Pixel Error kann sehr niedrige werte erzielen--> 4.0 .
    Ähnlich gute Ergebnisse erhielten wir durch alternierende Kernel Größen der initialen 5 Layer (3,5,3,5,3). Der Pixel Error war im Durschnitt
    bei 5.0 und konnte auch 4.3 erreichen. Hier scheinen die alternierenden Größen der Faltungskernel einen positiven Effekt auch das Erlernen
    von Features zu haben. Ein Effekt, der sich bei unser besten Architektur auch zum Teil beobachten lässt.
    Das Ergebnis ist in optimization_3.pdf zu finden.


4.  Abschließend, haben wir noch eine eigene Evaluationsmethode implementiert, die wir final nur getestet haben, aber nicht weiter verwendet haben,
    weil all unsere Auswertungen auf Basis der alten Funktion statt gefunden haben.




Generell haben wir die Netze immer mehrere male trainiert und nur das Beste Ergebnis gespeichert. Zur besseren
Vergleichbarkeit hätte man hier gut seinen Seed einsetzen können. Ebenfalls glauben wir, dass man ebenfalls eine bessere
Performance erreicht hätte, indem man nicht nur mehr patches für die Evalutation nimmt, sondern auch noch N-CNN´s trainiert
und deren Output später zusammen fügt.

Zur Implementierung in C++, hier haben wir uns auf den Kernteil beschränkt und nicht noch einmal alle Variationen 1-3 auch dort umgesetzt. Ebenfalls steht in c++ nur unsere Evaluationsmethode zur Verfügung.