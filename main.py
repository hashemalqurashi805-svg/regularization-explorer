import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import warnings

# إخفاء التحذيرات المزعجة (Future Warnings) ليكون التيرمينال نظيفاً
warnings.filterwarnings("ignore")

# 1. تجهيز البيانات
try:
    # محاولة تحميل ملف بيانات الاتصالات الفعلي
    df = pd.read_csv('communications_data.csv') 
    X = df.drop('target', axis=1) # تأكد أن اسم العمود هو target
    y = df['target']
    print("✅ تم تحميل البيانات بنجاح من الملف المحلي.")
except FileNotFoundError:
    print("⚠️ ملف CSV غير موجود. جاري توليد بيانات تجريبية لمحاكاة بيانات الاتصالات...")
    # توليد 200 عينة مع 10 ميزات لمحاكاة البيانات
    X_raw, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
    feature_names = [f'feature_{i}' for i in range(10)]
    X = pd.DataFrame(X_raw, columns=feature_names)

# 2. تقييس البيانات (Scaling)
# ضروري جداً لجعل قيم المعاملات قابلة للمقارنة تحت التنظيم
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. توليد 20 قيمة لـ C بشكل لوغاريتمي (من 0.001 إلى 100)
c_values = np.logspace(-3, 2, 20)

# 4. تدريب النماذج وتخزين المعاملات (Coefficients) لـ L1 و L2
l1_coefs = []
l2_coefs = []

print("🚀 جاري تدريب النماذج وحساب مسارات التنظيم...")

for c in c_values:
    # L1 (Lasso) - يستخدم لاختيار الميزات
    m1 = LogisticRegression(penalty='l1', C=c, solver='liblinear', max_iter=1000)
    m1.fit(X_scaled, y)
    l1_coefs.append(m1.coef_[0])
    
    # L2 (Ridge) - يستخدم لتقليص الأوزان بدون حذف الميزات
    m2 = LogisticRegression(penalty='l2', C=c, solver='liblinear', max_iter=1000)
    m2.fit(X_scaled, y)
    l2_coefs.append(m2.coef_[0])

l1_coefs = np.array(l1_coefs)
l2_coefs = np.array(l2_coefs)

# 5. رسم المخططات (Visualization)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# رسم مسار L1
for i, col in enumerate(X.columns):
    ax1.plot(c_values, l1_coefs[:, i], label=col)

ax1.set_xscale('log')
ax1.set_xlabel('C (Inverse Regularization Strength)')
ax1.set_ylabel('Coefficients Weights')
ax1.set_title('L1 Regularization Path (Lasso)\nNotice features hitting zero')
ax1.grid(True, linestyle='--', alpha=0.7)

# رسم مسار L2
for i, col in enumerate(X.columns):
    ax2.plot(c_values, l2_coefs[:, i], label=col)

ax2.set_xscale('log')
ax2.set_xlabel('C (Inverse Regularization Strength)')
ax2.set_title('L2 Regularization Path (Ridge)\nNotice smooth shrinking')
ax2.grid(True, linestyle='--', alpha=0.7)

# إضافة قائمة الميزات (Legend)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout()

# 6. حفظ الصورة لاستخدامها في GitHub
plt.savefig('regularization_comparison.png')
print("📂 تم حفظ المخطط بنجاح باسم 'regularization_comparison.png'")

# عرض الرسمة
plt.show()