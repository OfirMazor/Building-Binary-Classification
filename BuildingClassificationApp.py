import ee
import geemap
geemap.ee_initialize()
import pandas as pd
import streamlit as st
from plotly.express import line, bar


headContainer = st.container()
dataContainer = st.container()
paramsContainer = st.container()
resultsContainer = st.container()
visContainer = st.container()
st.set_page_config(layout = 'wide')




with headContainer:
    st.title('Extract Building Pixels with Binary Classification')
    st.text('This page describe 3 different models for supervised binary classifiction of buildings pixels from Sentinel2 imagery over Herzeliya city, Israel.')
    st.text('Models defined with Google Earth Engine library: https://earthengine.google.com')
    st.text('Maps visualization: https://geemap.org')     

            
            
HerzeliyaBorder = ee.FeatureCollection('users/emazorofir/BuildingClassification/HerzeliyaBorder')

sample_points = ee.FeatureCollection('users/emazorofir/BuildingClassification/HerzeliyaSamplePoints')

train_image = ee.ImageCollection('COPERNICUS/S2_SR') \
    .filterBounds(HerzeliyaBorder) \
    .filterDate('2021-04-01', '2021-09-30') \
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 0.5) \
    .mean() \
    .clip(HerzeliyaBorder)

pointMap = geemap.Map(center=[32.150, 34.816], zoom=13.4, add_google_map=True)
pointMap.addLayer(train_image, {'min': 0.0,'max': 10000.0,'bands': ['B4','B3','B2']}, name='Sentinel 2 True colors', opacity=0.95)            
pointMap.add_styled_vector(sample_points, column="ClassValue", palette = ['000000', 'F4F4F4'], layer_name="Sample points")

with dataContainer:
    st.subheader('Data ')
    st.text('- Buildings location:')
    st.markdown('https://www.openstreetmap.org')
    st.text('- Imagery: ')
    st.markdown('https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR')
    pointMap.to_streamlit(width=1000, height=500)
    st.text('White points: Building sample')
    st.text('Black points: None building sample')
    st.text('')
            
            
    
            
bands = ['B12','B8','B7','B6','B5','B4','B3','B2']
y = 'ClassValue'
X = train_image.select(bands).sampleRegions(**{
    'collection': sample_points,
    'properties': [y],
    'scale': 1}) \
    .randomColumn()  




with paramsContainer:
            st.text('')
            st.text('')
            st.subheader('Parametrs Definition:')
            st.text('')
            st.text('Gradient Tree Boost parameters')
            st.markdown('https://developers.google.com/earth-engine/apidocs/ee-classifier-smilegradienttreeboost')
            numberOfTrees = st.slider('Set number of trees:', 3, 400, 100, step=1)
            st.text('')
            st.text('CART classifier parameters')
            st.markdown('https://developers.google.com/earth-engine/apidocs/ee-classifier-smilecart')
            maxNodes = st.slider('Set maximum number of leaf nodes in each tree:', 3, 1000, 15, step=1)
            st.text('')
            st.text('Random Forest classifier parameters')
            st.markdown('https://developers.google.com/earth-engine/apidocs/ee-classifier-smilerandomforest')
            numberOfRandomTrees = st.slider('Set number of trees:', 3, 300, 50, step=1)
            st.text('')
        
smileGradientTreeBoost_clf = ee.Classifier.smileGradientTreeBoost(numberOfTrees=numberOfTrees).setOutputMode('CLASSIFICATION').train(X, y, bands)
smileGradientTreeBoost_train_image = train_image.select(bands).classify(classifier = smileGradientTreeBoost_clf).rename(['trainClass'])        

smileCart_clf = ee.Classifier.smileCart(maxNodes=maxNodes).setOutputMode('CLASSIFICATION').train(X, y, inputProperties = bands)
smileCart_train_image = train_image.select(bands).classify(classifier = smileCart_clf).rename(['trainClass'])
            
RandomForest_clf = ee.Classifier.smileRandomForest(numberOfTrees=numberOfRandomTrees).setOutputMode('CLASSIFICATION').train(X, y, bands)
RandomForest_train_image = train_image.select(bands).classify(classifier = RandomForest_clf).rename(['trainClass'])
    


    
def results_df(classifier):
    results = ee.Classifier.explain(classifier)
    keys = results.keys().getInfo()
    vals = results.values().getInfo()
    try:
        df = pd.DataFrame(vals[1].items(), columns=['band', 'importance']) #Random Forest importance feature location
        df['band'] = df['band'].str.replace('B', '').astype('int')
        df.sort_values(by='band', inplace=True)
    except:
        df = pd.DataFrame(vals[2].items(), columns=['band', 'importance']) #Smile CART importance feature location
        df['band'] = df['band'].str.replace('B', '').astype('int')
        df.sort_values(by='band', inplace=True)
    return df


smileCart_df = results_df(smileCart_clf)
smileGradientTreeBoosts_df = results_df(smileGradientTreeBoost_clf)
RandomForest_df = results_df(RandomForest_clf)

cart_IMPORTANCE_PLOT = line(smileCart_df, x="band", y="importance", width=1000, title = 'CART').update_traces(line=dict(color="orange", width=8))
gtb_IMPORTANCE_PLOT = line(smileGradientTreeBoosts_df, x="band", y="importance", width=1000, title = 'Gradient Tree Boosts').update_traces(line=dict(color="orange", width=8))
rf_IMPORTANCE_PLOT = line(RandomForest_df, x="band", y="importance", width=1000, title = 'Random Forest').update_traces(line=dict(color="orange", width=8))


models = [smileCart_clf, smileGradientTreeBoost_clf, RandomForest_clf]
models_name = ['CART', 'Gradient Tree Boost', 'Random Forest']

accuracy_values = []
consumersAccuracy_values = []
for m in models:
    accuracy_values.append(m.confusionMatrix().accuracy().getInfo())
    consumersAccuracy_values.append(m.confusionMatrix().consumersAccuracy().getInfo())
    
accuracy_df = pd.DataFrame(data=[accuracy_values, consumersAccuracy_values], columns=[models_name]).T
accuracy_df.reset_index(inplace=True)
accuracy_df.columns = ['Model','Accuracy', 'Consumers Accuracy']
accuracyPlot = bar(accuracy_df, x="Model", y="Accuracy", width=1000).update_traces(marker_color="cyan")

with resultsContainer:
    st.text('')
    st.subheader('Review Results')
    st.text('')
    st.text('1. Bands Importance in Model')
    st.write(cart_IMPORTANCE_PLOT)
    st.write(gtb_IMPORTANCE_PLOT)
    st.write(rf_IMPORTANCE_PLOT)
    st.text('2. Models accuracy')
    st.write(accuracy_df)
    st.write(accuracyPlot)
    
sentinel2_vis_params = {'min': 0.0,'max': 10000.0,'bands': ['B4','B3','B2']}
classes_vis_params = {'min' : 0, 'max' : 1, 'palette' : ['000000', 'F4F4F4']}

classifiedMap = geemap.Map(center=[32.150, 34.816], zoom=13.4, add_google_map=True)
classifiedMap.addLayer(train_image, {'min': 0.0,'max': 10000.0,'bands': ['B4','B3','B2']}, name='Sentinel 2 True colors', opacity=1.0)
classifiedMap.addLayer(smileGradientTreeBoost_train_image,classes_vis_params, name= 'Gradient Tree Boost')
classifiedMap.addLayer(smileCart_train_image,classes_vis_params, name= 'CART')
classifiedMap.addLayer(RandomForest_train_image,classes_vis_params, name= 'Random Forest')
classifiedMap.add_legend(legend_title='Legend', layer_name=smileGradientTreeBoost_train_image, legend_keys=['Building Class', 'None-Building Class'],legend_colors=['000000','F4F4F4'])

with visContainer:
    st.subheader('Classifiers Map')
    st.text('Explore the classified images with layres icon')
    classifiedMap.to_streamlit(width=1000, height=500)
