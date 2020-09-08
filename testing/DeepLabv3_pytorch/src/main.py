from utilities.labels import CityscapesLabels

city_labels = CityscapesLabels()
labels = city_labels.labels()
# DEBUG
print(len(labels))
print(city_labels.groups())




