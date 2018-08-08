from sklearn.datasets import load_iris
import graphviz 
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data) 
graph.render("iris") 

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)  
graph = graphviz.Source(dot_data)  
print(graph)


x = [[0.95374449,0.0154185, 0.030837] [0.0052592, 0.98797896, 0.00676183] [0.05672269, 0.04201681, 0.9012605 ]]

y = 
[[0.91611479 0.01324503 0.07064018]
 [0.00443131 0.99261448 0.00295421]
 [0.030837   0.05286344 0.91629956]]

z = 
[[0.97350993 0.02649007 0.        ]
 [0.0023622  0.9976378  0.        ]
 [0.10037175 0.05947955 0.8401487 ]]