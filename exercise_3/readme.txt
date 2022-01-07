Solution by: Falco Lentzsch(685454),Konrad von Kuegelgen(676609), Jesse Kruse(675710)

Im folgendene eine Kurze Beschreibung unserer Skripte.

python/Ex3.ipynb
--> Dieses Notebook enthält die Lösungen von Aufgabe 1-3

python/Ex+Bonus.ipynb
--> Hier sind ebenfalls Aufgabe 1-3 enthalten allerdings muss für Aufgabe 2 etwas an den Indizes verändert werden wie auch beschrieben.
	Wir haben einfach etwas unterschiedlich gerabeitet und deswegen auch zwei etwas abweichenden Lösungen mit gleichem Ergebnis.
	Das es jedoch hier etwas umständlich ist Aufgabe 2 auszuführen. Bitte das vorherige Notebook verwenden.
	Allerdings enthält dieses Notebook die BonusAufgabe, wofür Aufgabe 1-3 genutzt werden. Leider haben wir ein etwas abweichendes Ergebnis.
	Daher haben wir alle einzelnen Bilder der Kosten gedreht und ungedreht einmal mit ausgegeben. Wir wissen nicht, wo unser Fehler liegt

cpp/fvmbv_exercise3.cpp
--> Enthält unsere C++ Implementierung von Aufgaber 1-3 ohne Bonus.


Schriftlicher Teil 200 Words:
Zunächst einmal bieten die Verfahren mit "discrete motion" den Vorteil, dass wir nicht auf kleine Bewegungen limitiert sind, wie zum Beispiel
bei verfahren wie Optical Flow. Ebenfalls können wir bei Dynamic Programming selbest bestimmen wie unser Regularisierer aussehen soll und wie stark er zum Beispiel große Labelwechsel bestrafen soll. Ein weiterer Vorteil ist, das die Kostenfunktion/Regularisierer nicht differenzierbar sein muss und wir im Gegensatz zu Optical Flow die Möglichkeit haben einen Regularisierer zu verwenden, der nicht nur lokal sondern global regularisiert.
Ein Nachteil ist, dass wir uns zumindest beim Scanline Ansatz auf eine Verschiebungsrichtung beschränkt sind bzw. 2 (mit Rückrichtung). Das lässt sich erweitern durch Ansätze wie SGM, jedoch wird das ganze wesentlich größer und Rechenintensiever. Die Berechnungskomplexität bleibt jedoch immernoch vergleichsweise gering.
Ein Nachteil des ganzen ist jedoch, da wir hier mit Rekursionen arbeiten, lässt sich das ganze nicht völlig parallelisieren. Ein weiterer Nachteil ist, dass wir außerdem vorab unseren Suchraum festlegen müssen und dieser darüber bestimmt, wie gut am Ende unsere lösung aussieht.
