#!/usr/bin/env python
# coding: utf-8

# # **Import Library**

# !pip install imblearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf
# # **Load Dataset**

# In[3]:


bankchurn = pd.read_csv('./BankChurners.csv')
bankchurn = bankchurn.drop(['CLIENTNUM'], axis = 1)
bankchurn.head()


# # **Exploratory Data Analysis**

# ## **Descriptive Statistics**

# In[4]:


bankchurn.describe()


# In[5]:


bankchurn.describe(include = 'O')


# ## **Proportion of Target Class**

# In[6]:


target = bankchurn["Attrition_Flag"].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(target, labels = target.index, autopct = '%1.1f%%', shadow = True, explode = [0.1, 0])
ax1.axis('equal')
plt.title("Amount of churned customers", fontsize = 14)
plt.show()
print('Total number of customers:', bankchurn['Attrition_Flag'].count())
print(bankchurn['Attrition_Flag'].value_counts())


# ## **Numerical Variables by Target**

# In[7]:


# Customer Age
plt.figure()
sns.distplot(bankchurn.Customer_Age[bankchurn.Attrition_Flag == 'Attrited Customer'], bins = [26, 35, 45, 55, 75], color = 'r', label = 'Attrited Customer')
sns.distplot(bankchurn.Customer_Age[bankchurn.Attrition_Flag == 'Existing Customer'], bins = [26, 35, 45, 55, 75], color = 'g', label = 'Existing Customer')
plt.title('Customer Age\n', fontsize = 16, fontweight = 'semibold')
plt.legend()
plt.show()


# In[8]:


# Dependent_count
plt.figure()
sns.distplot(bankchurn.Dependent_count[bankchurn.Attrition_Flag == 'Attrited Customer'], color = 'r', label = 'Attrited Customer')
sns.distplot(bankchurn.Dependent_count[bankchurn.Attrition_Flag == 'Existing Customer'], color = 'g', label = 'Existing Customer')
plt.title('Dependent Count\n', fontsize = 16, fontweight = 'semibold')
plt.legend()
plt.show()



# In[9]:


# Months_on_book
plt.figure()
sns.distplot(bankchurn.Months_on_book[bankchurn.Attrition_Flag == 'Attrited Customer'], bins = 12, color = 'r', label = 'Attrited Customer')
sns.distplot(bankchurn.Months_on_book[bankchurn.Attrition_Flag == 'Existing Customer'], bins = 12, color = 'g', label = 'Existing Customer')
plt.title('Month on Book\n', fontsize = 16, fontweight = 'semibold')
plt.legend()
plt.show()


# In[10]:


# Total_Relationship_Count
plt.figure()
sns.distplot(bankchurn.Total_Relationship_Count[bankchurn.Attrition_Flag == 'Attrited Customer'], color = 'r', label = 'Attrited Customer')
sns.distplot(bankchurn.Total_Relationship_Count[bankchurn.Attrition_Flag == 'Existing Customer'], color = 'g', label = 'Existing Customer')
plt.title('Total Relationship Count\n', fontsize = 16, fontweight = 'semibold')
plt.legend()
plt.show()


# In[11]:


# Months_Inactive_12_mon
plt.figure()
sns.distplot(bankchurn.Months_Inactive_12_mon[bankchurn.Attrition_Flag == 'Attrited Customer'], color = 'r', label = 'Attrited Customer')
sns.distplot(bankchurn.Months_Inactive_12_mon[bankchurn.Attrition_Flag == 'Existing Customer'], color = 'g', label = 'Existing Customer')
plt.title('Months Inactive 12 mon\n', fontsize = 16, fontweight = 'semibold')
plt.legend()
plt.show()


# In[12]:


# Contacts_Count_12_mon
plt.figure()
sns.displot(bankchurn.Contacts_Count_12_mon[bankchurn.Attrition_Flag == 'Attrited Customer'], color = 'r', label = 'Attrited Customer')
sns.displot(bankchurn.Contacts_Count_12_mon[bankchurn.Attrition_Flag == 'Existing Customer'], color = 'g', label = 'Existing Customer')
plt.title('Contacts Count 12 mon\n', fontsize = 16, fontweight = 'semibold')
plt.legend()
plt.show()


# In[13]:


# Credit_Limit
plt.figure()
sns.distplot(bankchurn.Credit_Limit[bankchurn.Attrition_Flag == 'Attrited Customer'], bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000], color = 'r', label = 'Attrited Customer')
sns.distplot(bankchurn.Credit_Limit[bankchurn.Attrition_Flag == 'Existing Customer'], bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000], color = 'g', label = 'Existing Customer')
plt.title('Credit Limit\n', fontsize = 16, fontweight = 'semibold')
plt.legend()
plt.show()


# In[14]:


# Total_Revolving_Bal
plt.figure()
sns.distplot(bankchurn.Total_Revolving_Bal[bankchurn.Attrition_Flag == 'Attrited Customer'], bins = [0, 500, 1000, 1500, 2000, 2500, 3000], color = 'r', label = 'Attrited Customer')
sns.distplot(bankchurn.Total_Revolving_Bal[bankchurn.Attrition_Flag == 'Existing Customer'], bins = [0, 500, 1000, 1500, 2000, 2500, 3000], color = 'g', label = 'Existing Customer')
plt.title('Total Revolving Balance\n', fontsize = 16, fontweight = 'semibold')
plt.legend()
plt.show()


# In[15]:


# Avg_Open_To_Buy
plt.figure()
sns.distplot(bankchurn.Avg_Open_To_Buy[bankchurn.Attrition_Flag == 'Attrited Customer'], color = 'r', label = 'Attrited Customer')
sns.distplot(bankchurn.Avg_Open_To_Buy[bankchurn.Attrition_Flag == 'Existing Customer'], color = 'g', label = 'Existing Customer')
plt.title('Average Open to Buy\n', fontsize = 16, fontweight = 'semibold')
plt.legend()
plt.show()


# In[16]:


# Total_Amt_Chng_Q4_Q1
plt.figure()
sns.distplot(bankchurn.Total_Amt_Chng_Q4_Q1[bankchurn.Attrition_Flag == 'Attrited Customer'], color = 'r', label = 'Attrited Customer')
sns.distplot(bankchurn.Total_Amt_Chng_Q4_Q1[bankchurn.Attrition_Flag == 'Existing Customer'], color = 'g', label = 'Existing Customer')
plt.title('Total Amt Chng Q4 Q1\n', fontsize = 16, fontweight = 'semibold')
plt.legend()
plt.show()


# In[17]:


# Total_Trans_Amt
plt.figure()
sns.distplot(bankchurn.Total_Trans_Amt[bankchurn.Attrition_Flag == 'Attrited Customer'], color = 'r', label = 'Attrited Customer')
sns.distplot(bankchurn.Total_Trans_Amt[bankchurn.Attrition_Flag == 'Existing Customer'], color = 'g', label = 'Existing Customer')
plt.title('Total Trans Amt\n', fontsize = 16, fontweight = 'semibold')
plt.legend()
plt.show()


# In[18]:


# Total_Trans_Ct
plt.figure()
sns.distplot(bankchurn.Total_Trans_Ct[bankchurn.Attrition_Flag == 'Attrited Customer'], bins = 10, color = 'r', label = 'Attrited Customer')
sns.distplot(bankchurn.Total_Trans_Ct[bankchurn.Attrition_Flag == 'Existing Customer'], bins = 10, color = 'g', label = 'Existing Customer')
plt.title('Total Trans Ct\n', fontsize = 16, fontweight = 'semibold')
plt.legend()
plt.show()


# In[19]:


# Total_Ct_Chng_Q4_Q1
plt.figure()
sns.distplot(bankchurn.Total_Ct_Chng_Q4_Q1[bankchurn.Attrition_Flag == 'Attrited Customer'], color = 'r', label = 'Attrited Customer')
sns.distplot(bankchurn.Total_Ct_Chng_Q4_Q1[bankchurn.Attrition_Flag == 'Existing Customer'], color = 'g', label = 'Existing Customer')
plt.title('Total Ct Chng Q4 Q1\n', fontsize = 16, fontweight = 'semibold')
plt.legend()
plt.show()


# In[20]:


# Avg_Utilization_Ratio
plt.figure()
sns.distplot(bankchurn.Avg_Utilization_Ratio[bankchurn.Attrition_Flag == 'Attrited Customer'], color = 'r', label = 'Attrited Customer')
sns.distplot(bankchurn.Avg_Utilization_Ratio[bankchurn.Attrition_Flag == 'Existing Customer'], color = 'g', label = 'Existing Customer')
plt.title('Avg Utilization Ratio\n', fontsize = 16, fontweight = 'semibold')
plt.legend()
plt.show()


# ## **Categorical Variables**

# ### **Gender**

# In[21]:


gender = bankchurn["Gender"].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(gender, labels = gender.index, autopct = '%1.1f%%', shadow = True, explode = [0.1, 0])
ax1.axis('equal')
plt.title("Amount of Customer's Gender", fontsize = 14)
plt.show()
print('Total number of customers:', bankchurn['Gender'].count())
print(bankchurn['Gender'].value_counts())


# ### **Education Level**

# In[22]:


proportion_education = bankchurn['Education_Level'].value_counts()
proportion_education = pd.DataFrame(proportion_education)
proportion_education = proportion_education.rename(columns={'Education_Level':'Count'})
proportion_education = proportion_education.rename_axis('Education_Level').reset_index()
proportion_education


# In[23]:


proportion_education.head()


# In[24]:


# Education_Level
# Show the proportion of education in bar chart
plt.figure(figsize = (15, 7))

percentage = []
for i in proportion_education['count']:
  pct = (i / proportion_education['count'].sum()) * 100
  percentage.append(round(pct, 2))
proportion_education['Percentage'] = percentage

## Show the plot
plots = sns.barplot(x = "Education_Level",
                    y = "Percentage",
                    data = proportion_education)

## Show the annotation
for p in plots.patches:
  plots.annotate('{} %'.format(p.get_height().astype('float')),
                 (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha='center', va='center',
                 size=15, xytext=(0, 8),
                 textcoords='offset points')

# Setting the label for x-axis
plt.xlabel("Education Level", size=14)
# Setting the label for y-axis
plt.ylabel("Percentage", size=14)
# Setting the title for the graph
plt.title("Percentage of Education Level", size = 16, weight = 'semibold')
# Fianlly showing the plot
plt.show()


# In[25]:


proportion_education = bankchurn['Education_Level'].value_counts()
proportion_education = pd.DataFrame(proportion_education)
proportion_education = proportion_education.rename(columns={'Education_Level':'Count'})
proportion_education = proportion_education.rename_axis('Education_Level').reset_index()
proportion_education


# ### **Marital Status**

# In[26]:


proportion_marital_status = bankchurn['Marital_Status'].value_counts()
proportion_marital_status = pd.DataFrame(proportion_marital_status)
proportion_marital_status = proportion_marital_status.rename(columns={'Marital_Status':'Count'})
proportion_marital_status = proportion_marital_status.rename_axis('Marital_Status').reset_index()
proportion_marital_status


# In[27]:


# Show the proportion of education in bar chart
plt.figure(figsize = (15, 7))

percentage = []
for i in proportion_marital_status['count']:
  pct = (i / proportion_marital_status['count'].sum()) * 100
  percentage.append(round(pct, 2))
proportion_marital_status['Percentage'] = percentage

## Show the plot
plots = sns.barplot(x = "Marital_Status",
                    y = "Percentage",
                    data = proportion_marital_status)

## Show the annotation
for p in plots.patches:
  plots.annotate('{} %'.format(p.get_height().astype('float')),
                 (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha='center', va='center',
                 size=15, xytext=(0, 8),
                 textcoords='offset points')

# Setting the label for x-axis
plt.xlabel("Marital Status", size=14)
# Setting the label for y-axis
plt.ylabel("Percentage", size=14)
# Setting the title for the graph
plt.title("Percentage of Marital Status", size = 16, weight = 'semibold')
# Fianlly showing the plot
plt.show()


# ### **Income Category**

# In[28]:


proportion_income_category = bankchurn['Income_Category'].value_counts()
proportion_income_category = pd.DataFrame(proportion_income_category)
proportion_income_category = proportion_income_category.rename(columns={'Income_Category':'Count'})
proportion_income_category = proportion_income_category.rename_axis('Income_Category').reset_index()
proportion_income_category


# In[29]:


# Show the proportion of education in bar chart
plt.figure(figsize = (15, 7))

percentage = []
for i in proportion_income_category['count']:
  pct = (i / proportion_income_category['count'].sum()) * 100
  percentage.append(round(pct, 2))
proportion_income_category['Percentage'] = percentage

## Show the plot
plots = sns.barplot(x = "Income_Category",
                    y = "Percentage",
                    data = proportion_income_category)

## Show the annotation
for p in plots.patches:
  plots.annotate('{} %'.format(p.get_height().astype('float')),
                 (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha='center', va='center',
                 size=15, xytext=(0, 8),
                 textcoords='offset points')

# Setting the label for x-axis
plt.xlabel("Income Category", size=14)
# Setting the label for y-axis
plt.ylabel("Percentage", size=14)
# Setting the title for the graph
plt.title("Percentage of Income Category", size = 16, weight = 'semibold')
# Fianlly showing the plot
plt.show()


# ### **Card Category**

# In[30]:


proportion_card_category = bankchurn['Card_Category'].value_counts()
proportion_card_category = pd.DataFrame(proportion_card_category)
proportion_card_category = proportion_card_category.rename(columns={'Card_Category':'Count'})
proportion_card_category = proportion_card_category.rename_axis('Card_Category').reset_index()
proportion_card_category


# In[31]:


# Show the proportion of education in bar chart
plt.figure(figsize = (15, 7))

percentage = []
for i in proportion_card_category['count']:
  pct = (i / proportion_card_category['count'].sum()) * 100
  percentage.append(round(pct, 2))
proportion_card_category['Percentage'] = percentage

## Show the plot
plots = sns.barplot(x = "Card_Category",
                    y = "Percentage",
                    data = proportion_card_category)

## Show the annotation
for p in plots.patches:
  plots.annotate('{} %'.format(p.get_height().astype('float')),
                 (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha='center', va='center',
                 size=15, xytext=(0, 8),
                 textcoords='offset points')

# Setting the label for x-axis
plt.xlabel("Card Category", size=14)
# Setting the label for y-axis
plt.ylabel("Percentage", size=14)
# Setting the title for the graph
plt.title("Percentage of Card Category", size = 16, weight = 'semibold')
# Fianlly showing the plot
plt.show()


# ## **Checking Outliers**

# In[32]:


dfo1 = bankchurn.select_dtypes(include = ['int64'])
plt.figure(figsize = (20, 7))
sns.boxplot(x = 'variable', y = 'value', color = 'green', orient = 'v', data = pd.melt(dfo1))
plt.tight_layout()


# In[33]:


dfo2 = bankchurn.select_dtypes(include = ['float64']) # memilih kolom numerik
plt.figure(figsize = (20, 7))
sns.boxplot(x = 'variable', y = 'value', color = 'green', orient = 'v', data = pd.melt(dfo2))
plt.tight_layout()


# ## **Checking Correlation**

# In[34]:


dfk = bankchurn.select_dtypes(include = ['int64', 'float64']) # memilih kolom numerik
k = len(dfk.columns)
cm = dfk.corr()
plt.figure(figsize = (14, 10))
sns.heatmap(cm, annot = True, cmap = 'viridis')


# ## **Normality Test**

# In[35]:


numeric_uji = bankchurn[['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
                        'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']].sample(n = 50, random_state = 21)
numeric_uji.head()


# In[36]:


nilai_statistik = list()
p_value = list()
hasil = list()
for i in numeric_uji:
    ns, pv = stats.shapiro(numeric_uji[i])
    hsl = 'normal' if pv > .05 else 'tidak normal'
    nilai_statistik.append(ns)
    p_value.append(pv)
    hasil.append(hsl)

uji = pd.DataFrame(nilai_statistik, index = numeric_uji.columns, columns = ['nilai statistik'])
uji['p-value'] = p_value
uji['sig lvl'] = 0.05
uji['hasil uji'] = hasil
uji


# # **Data Preprocessing**

# ## **Remove Credit Limit**

# In[37]:


bankchurn = bankchurn.drop('Credit_Limit', axis = 1)
bankchurn.info()


# In[38]:


bankchurn.head()


# ## **Feature Scalling**

# In[39]:


stand_var = ['Customer_Age', 'Months_on_book', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1']
norm_var = ['Dependent_count', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
            'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Avg_Utilization_Ratio']

for stand in bankchurn[stand_var]:
    bankchurn[stand] = StandardScaler().fit_transform(bankchurn[stand].values.reshape(len(bankchurn), 1))
for norm in bankchurn[norm_var]:
    bankchurn[norm] = MinMaxScaler().fit_transform(bankchurn[norm].values.reshape(len(bankchurn), 1))


# In[40]:


bankchurn.head()


# ## **Categorical Encoding**

# In[41]:


# Ordinal Encoding
LE = LabelEncoder()
for cat in list(['Attrition_Flag', 'Education_Level', 'Income_Category', 'Card_Category']):
    bankchurn[cat] = LE.fit_transform(bankchurn[cat])

# Nominal Encoding
nominal_cats = ['Gender', 'Marital_Status']
for cat in nominal_cats:
    onehot = pd.get_dummies(bankchurn[cat], prefix = cat)
    bankchurn = bankchurn.join(onehot)

bankchurn = bankchurn.drop(['Gender', 'Marital_Status'], axis = 1)


# In[42]:


feature_names = bankchurn.columns.tolist()


# ## **Train Test Split**

# In[43]:


X = bankchurn.drop(['Attrition_Flag'], axis = 1).astype(float).values
y = bankchurn['Attrition_Flag'].astype(float).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[44]:


print(X)


# ## **Imbalanced Dataset**

# In[45]:


smote = SMOTE(random_state = 0)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print('Before SMOTE')
print(pd.DataFrame(y_train).value_counts())
print('After SMOTE')
print(pd.DataFrame(y_resampled).value_counts())


# # **Modeling**

# ## **Logistic Regression**

# In[46]:


# Train model
log_model = LogisticRegression().fit(X_resampled, y_resampled)

# Predict
log_y_test_pred = log_model.predict(X_test)

# Performance
print('Accuracy:', accuracy_score(y_test, log_y_test_pred))
print('Precision:', precision_score(y_test, log_y_test_pred, average = 'macro'))
print('Recall/Sensitivity:', recall_score(y_test, log_y_test_pred, average = 'macro'))
print('Cross Entropy Loss:', tf.keras.metrics.binary_crossentropy(y_test, log_y_test_pred).numpy())


# In[47]:


# Form confusion matrix as a DataFrame
confusion_matrix_df = pd.DataFrame((confusion_matrix(y_test, log_y_test_pred)),
                                   ('Existing Customer', 'Atrited Customer'),
                                   ('Existing Customer', 'Atrited Customer'))

# Plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_df, annot = True, annot_kws = {'size' : 14}, fmt = 'd', cmap = 'YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0, ha = 'right', fontsize = 14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation = 0, ha = 'center', fontsize = 14)

plt.title('Confusion Matrix for Logistic Regression\n', fontsize = 18, color = 'darkblue')
plt.ylabel('True label', fontsize = 14)
plt.xlabel('Predicted label',fontsize = 14)
plt.show()


# ## **Random Forest Classifier**

# In[48]:


# Train model
rf_model = RandomForestClassifier(random_state = 42).fit(X_resampled, y_resampled)

# Predict
rf_y_test_pred = rf_model.predict(X_test)

# Performance
print('Accuracy:', accuracy_score(y_test, rf_y_test_pred))
print('Precision:', precision_score(y_test, rf_y_test_pred, average = 'macro'))
print('Recall/Sensitivity:', recall_score(y_test, rf_y_test_pred, average = 'macro'))
print('Cross Entropy Loss:', tf.keras.metrics.binary_crossentropy(y_test, rf_y_test_pred).numpy())


# In[49]:


# Form confusion matrix as a DataFrame
confusion_matrix_df = pd.DataFrame((confusion_matrix(y_test, rf_y_test_pred)),
                                   ('Existing Customer', 'Atrited Customer'),
                                   ('Existing Customer', 'Atrited Customer'))

# Plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_df, annot = True, annot_kws = {'size' : 14}, fmt = 'd', cmap = 'YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0, ha = 'right', fontsize = 14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation = 0, ha = 'center', fontsize = 14)

plt.title('Confusion Matrix for Random Forest Classifier\n', fontsize = 18, color = 'darkblue')
plt.ylabel('True label', fontsize = 14)
plt.xlabel('Predicted label', fontsize = 14)
plt.show()


# In[50]:


import shap
explainer_rf = shap.TreeExplainer(rf_model)

# Compute SHAP values
shap_values_rf = explainer_rf.shap_values(X_test)

# Check the shape of shap_values
print(f"SHAP values shape: {shap_values_rf.shape}")

# Plot SHAP summary plot
shap.summary_plot(shap_values_rf, X_test, feature_names=feature_names)
plt.title('SHAP Summary Plot for Random Forest Classifier')
plt.show()


# ## **Gradient Boosting Classifier**

# In[ ]:


# Train model
gb_model = GradientBoostingClassifier().fit(X_resampled, y_resampled)

# Predict
gb_y_test_pred = gb_model.predict(X_test)

# Performance
print('Accuracy:', accuracy_score(y_test, gb_y_test_pred))
print('Precision:', precision_score(y_test, gb_y_test_pred, average = 'macro'))
print('Recall/Sensitivity:', recall_score(y_test, gb_y_test_pred, average = 'macro'))
print('Cross Entropy Loss:', tf.keras.metrics.binary_crossentropy(y_test, gb_y_test_pred).numpy())


# In[ ]:


# Form confusion matrix as a DataFrame
confusion_matrix_df = pd.DataFrame((confusion_matrix(y_test, gb_y_test_pred)),
                                   ('Existing Customer', 'Atrited Customer'),
                                   ('Existing Customer', 'Atrited Customer'))

# Plot confusion matrix
plt.figure()
heatmap = sns.heatmap(confusion_matrix_df, annot = True, annot_kws = {'size' : 14}, fmt = 'd', cmap = 'YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0, ha = 'right', fontsize = 14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation = 0, ha = 'center', fontsize = 14)

plt.title('Confusion Matrix for Gradient Boosting Classifier\n', fontsize = 18, color = 'darkblue')
plt.ylabel('True label', fontsize = 14)
plt.xlabel('Predicted label', fontsize = 14)
plt.show()


# # **Feature Value Analytics**

# In[ ]:


import shap
explainer = shap.TreeExplainer(gb_model)
shap_values = explainer.shap_values(X_test)

# Check the shape of shap_values
print(f"SHAP values shape: {len(shap_values)}")


shap.summary_plot(shap_values, X_test,feature_names=feature_names)

