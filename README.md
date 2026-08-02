# 🚢 Titanic Survival Prediction System | نظام التنبؤ بالنجاة لركاب التيتانيك

An end-to-end Machine Learning project that compares multiple classification models, saves the best performing model pipeline (SVM), and deploys it as an interactive Streamlit web application with a modern, glassmorphism design.

مشروع متكامل لتعلم الآلة يقوم بالمقارنة بين عدة نماذج تصنيف مختلفة، واختيار النموذج الأفضل (SVM) وحفظه، ثم بناء تطبيق ويب تفاعلي وتصميم عصري باستخدام Streamlit للتنبؤ باحتمالية نجاة الركاب.

---

## 📂 Project Structure | هيكلية المشروع

* 📄 `train.csv`: Training dataset | بيانات التدريب الأساسية.
* 📄 `test.csv`: Testing dataset (raw data) | بيانات الاختبار لتوليد التوقعات.
* 📄 `yuosef_tarek_day16 (1).ipynb`: Initial Jupyter Notebook containing exploration and preprocessing | النوت بوك الخاص بالتحليل والاستكشاف الأولي.
* 🐍 `train_and_save.py`: Script to run cross-validation on 6 classification models, save the best one, and output test predictions | سكربت مقارنة النماذج وحفظ أفضل نموذج وتوليد ملف التوقعات.
* 🐍 `app.py`: Streamlit web application (Arabic, RTL, Custom Dark Theme) | تطبيق الويب التفاعلي بواجهة عربية وتصميم حديث.
* 📄 `best_titanic_model.joblib`: The saved trained SVM model pipeline | ملف خط المعالجة والنموذج المدرب المحفوظ.
* 📄 `predictions.csv`: Model predictions on the test dataset | ملف التوقعات لبيانات الاختبار.
* 📄 `requirements.txt`: Python package dependencies | ملف المكتبات المطلوبة لتشغيل المشروع.

---

## 🚀 Getting Started | كيف تبدأ

### 1. Installation | تثبيت المكتبات
Make sure you have Python installed, then install all the required libraries:
تأكد من تثبيت بايثون على جهازك، ثم قم بتثبيت المكتبات المطلوبة عبر الأمر التالي:
```bash
pip install -r requirements.txt
```

### 2. Model Training & Comparison | مقارنة وتدريب النماذج
Run the training script to evaluate the models, save the best one to `best_titanic_model.joblib`, and generate the `predictions.csv` file:
شغل سكربت التدريب للمقارنة بين 6 نماذج تصنيف مختلفة، وحفظ النموذج الأفضل وتوليد ملف التوقعات لبيانات الاختبار:
```bash
python train_and_save.py
```

#### Evaluation Results (5-Fold Cross Validation Accuracy):
#### نتائج تقييم النماذج (دقة التحقق المتقاطع):
* **Logistic Regression**: `79.35%`
* **Random Forest**: `82.26%`
* **Gradient Boosting**: `82.60%`
* **Support Vector Machine (SVM/SVC)**: **`82.71%`** 👑 *(Selected / النموذج المختار)*
* **K-Nearest Neighbors (KNN)**: `81.14%`
* **Decision Tree**: `78.90%`

### 3. Running the Web Application | تشغيل تطبيق الويب
Run the Streamlit server to open the interactive dashboard:
شغّل خادم Streamlit لفتح لوحة التحكم والتطبيق التفاعلي:
```bash
streamlit run app.py
```

---

## 🎨 UI Features | مميزات تطبيق الويب
* **Arabic & RTL Support**: Full Arabic interface aligned to the right (Cairo font included).
* **Modern Dark Mode**: Sleek linear gradient backgrounds with blue-indigo colors.
* **Glassmorphism Design**: Input cards with frosted glass effect and subtle borders.
* **Dynamic Result Cards**: Green/Emerald card for survival, Red/Crimson card for perish, including calculated probability percentages.

* **دعم كامل للغة العربية (RTL)**: واجهة مستخدم ناطقة بالكامل باللغة العربية مع خط Cairo.
* **تصميم داكن حديث**: خلفيات متدرجة مميزة بألوان جذابة وهادئة.
* **تأثير الزجاج البلوري (Glassmorphism)**: بطاقات إدخال شبه شفافة وتفاعلية.
* **عرض ديناميكي للنتائج**: بطاقات ملونة مميزة تتفاعل حسب التوقع (أخضر للنجاة، أحمر للوفاة) مع إظهار نسب الاحتمالية بدقة.
