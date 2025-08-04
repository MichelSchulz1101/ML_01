# Farben erkennen

## Was sollte mein Modell erkennen?
Ich habe mit dem Programm Teachable Machine ein ML-Modell zur Farberkennung erstellt. Es dient dazu, zwischen den Farben Rot und Blau in Bildern zu unterscheiden.

## Wie bin ich vorgegangen?
Zuerst habe ich zwei Klassen erstellt und sie *Rot* und *Blau* genannt. Danach habe ich mit meiner Webcam Bilder von roten Gegenständen aufgenommen und sie der Klasse *Rot* zugewiesen. Anschließend habe ich blaue Gegenstände aufgenommen und sie der Klasse *Blau* hinzugefügt. Danach konnte ich das Modell trainieren.

## Wie viele Klassen habe ich erstellt?
- 2

## Wie viele Bilder habe ich pro Klasse aufgenommen?
- 20

## Besonderheiten (z. B. schwierige Lichtverhältnisse, Wiederholungen, Abwechslung)?
Ich habe versucht, möglichst viele unterschiedliche Blau- und Rottöne zu verwenden. Außerdem habe ich die Objekte in unterschiedlichen Abständen zur Webcam gehalten, um mehr Variation zu erzeugen.

## Wie lange hat das Training ca. gedauert?
- Etwa 8 Sekunden

## Gab es dabei Fehlermeldungen oder technische Probleme?
- Nein

## Wie habe ich mein Modell getestet?
Zum Testen habe ich blaue oder rote Gegenstände vor die Webcam gehalten und überprüft, ob die Vorhersage korrekt war.

## Wie zuverlässig war das Modell meiner Einschätzung nach?
- Ungefähr 80 %

## Wurde eine Klasse häufiger falsch erkannt als die andere?
- Ja, *Rot* wurde häufiger falsch erkannt.

## Wie könnte man die Erkennungsqualität verbessern?
- Den Hintergrund ausblenden  
- Bilder bei optimalen Lichtverhältnissen aufnehmen