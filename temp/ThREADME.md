# Hessian Optimization Theory in Deep Learning

จัดทำโดย

นายจักรรินทร์ พรมสี 67070501009  
นายทีฆทัศน์ วงศ์สืบสันตติ 67070501019  
นายธวัฒน์ บุญสุข 67070501024  
นายธีรกาญจน์ น้อยรักษา 67070501062  
นายอชิรวิชร์ ประสม 67070501076  
นายพงศธร พุทธสอน 67070501084

เสนอ

ดร.อนุวัฒน์ ตั้งธนวัฒน์สกุล

รายงานประกอบวิชา **MTH 234 Linear Algebra**  
มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าธนบุรี  
ภาคการศึกษาที่ 1 ปีการศึกษา 2568

---

## Abstract

เมทริกซ์เฮสเซียน **H** คือเมทริกซ์สมมาตรจัตุรัสของอนุพันธ์ย่อยอันดับที่สองของฟังก์ชันค่าสเกลาร์หลายตัวแปร \(f(x)\) ซึ่งเป็นรากฐานของการวิเคราะห์ความโค้งเฉพาะที่ในปริภูมิหลายมิติ เมทริกซ์นี้ถูกใช้เป็นแกนหลักในการประมาณค่าฟังก์ชันรอบจุดวิกฤตให้อยู่ในรูปกำลังสอง _(Quadratic Form)_ ผ่านอนุกรมเทย์เลอร์

การวิเคราะห์คุณสมบัติทางคณิตศาสตร์ของ H มีความสำคัญดังนี้:

1. **ความเป็นสมมาตร** ทำให้ eigenvalues เป็นจำนวนจริง และช่วยให้การผกผันเมทริกซ์ \(H^{-1}\) มีเสถียรภาพ
2. **Eigenvectors** กำหนดทิศทางหลักของความโค้ง ส่วน **Eigenvalues** บอกขนาดของความโค้ง ซึ่งสามารถสร้าง _orthonormal basis_ เพื่อวิเคราะห์เชิงลึก
3. ผลคูณเมทริกซ์–เวกเตอร์ \(Hv\) ทำให้ประเมินผลกระทบของความโค้งในทิศทาง \(v\) ได้โดยตรง

องค์ประกอบทั้งหมดนี้ช่วยในการจำแนกจุดวิกฤต (local minima, maxima, saddle point) และเป็นหัวใจของอัลกอริทึม second-order เช่น Newton’s Method และ Quasi-Newton Methods

---

## Introduction

Linear Algebra เป็นพื้นฐานสำคัญของการคำนวณสมัยใหม่ ทั้งในงานประมวลผลข้อมูลหลายมิติ สัญญาณ การวิเคราะห์ข้อมูลขนาดใหญ่ และการออกแบบอัลกอริทึมต่าง ๆ โดยเฉพาะงานด้าน Deep Learning ซึ่งอาศัยโครงสร้างเชิงเส้นของเวกเตอร์ เมทริกซ์ และการแปลงเชิงเส้นอย่างมาก

หัวข้อหนึ่งที่โดดเด่นคือการวิเคราะห์ความโค้งของ loss function ผ่าน _Hessian Matrix_ เมทริกซ์นี้อธิบายโครงสร้างโค้งของ loss landscape และช่วยวิเคราะห์จุดต่ำสุด จุดอาน หรือจุดสูงสุด รวมถึงใช้เป็นพื้นฐานของ second-order optimization

รายงานนี้มุ่งอธิบายบทบาทของ Linear Algebra ใน Deep Learning และความสำคัญของ Hessian ต่อการวิเคราะห์และเพิ่มประสิทธิภาพการเทรน

---

## Mathematical Concepts from Linear Algebra

- **Matrix-Vector Product**: ใช้ประเมินผลของ \(H\) ต่อเวกเตอร์ \(v\) โดยไม่ต้องสร้าง H
- **Orthogonal / Orthonormal Basis**: สำคัญต่อการทำ eigen decomposition
- **Eigenvalues / Eigenvectors**: ทิศทางและขนาดของความโค้งของ Hessian
- **Eigen Basis**: สร้างฐานตั้งฉากจาก eigenvectors
- **Hessian Matrix (H)**: เมทริกซ์สมมาตรของอนุพันธ์อันดับสอง
- **Linear Transformation**: มอง Hessian เป็นการแปลงเชิงเส้น
- **Taylor Series**: ใช้ประมาณ loss ด้วย quadratic form

---

# Detail Description

## 1. หลักการทำงานของ Deep Learning ทั่วไป

### 1.1 โครงสร้างการเชื่อมต่อของ Neural Network และ Non-linear Relation

Neural network ประกอบด้วย layer หลายชั้น โดยแต่ละชั้นมี neurons หลายตัว ซึ่งเชื่อมโยงกันด้วย weights และ bias  
การเชื่อมต่อระหว่าง neuron ทำให้ระบบสามารถสร้าง non-linear mapping จาก input ไปยัง output ได้  
การใช้ activation function เช่น **ReLU**, **Sigmoid**, หรือ **Tanh** เป็นปัจจัยสำคัญที่ทำให้เครือข่ายสามารถสร้างความสัมพันธ์แบบไม่เชิงเส้น (non-linear relation)  
การเชื่อมต่อหลายชั้น (deep layers) ทำให้โมเดลสามารถเรียนรู้ความซับซ้อนของปัญหาได้

**รูปที่ 1: โครงสร้างและการเชื่อมต่อของเซลล์ประสาทเทียม (ที่มา: GeeksforGeeks [1])**

---

### 1.2 การคำนวณ Forward Pass

Forward pass คือการคำนวณผลลัพธ์ของโมเดลจาก input ผ่านทุก neurons และ layers ทำให้เราได้การทำนายของทั้งโมเดล ซึ่งใช้คำนวณ loss เพื่อการทำ gradient ในหัวข้อถัดไป

**รูปที่ 2: โครงสร้างและการเชื่อมต่อของเซลล์ประสาทเทียม (ที่มา: GeeksforGeeks [2])**

---

#### 1.2.1 การคำนวณ Affine Transformation

ในแต่ละ layer \( l \) จะคำนวณผลลัพธ์จากการแปลงเชิงเส้น (Affine):

\[
z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}
\]

โดยที่

- \( a^{(l-1)} \) คือ activation จาก layer ก่อนหน้า
- \( W^{(l)} \) คือ weight matrix
- \( b^{(l)} \) คือ bias vector

---

#### 1.2.2 การคำนวณ Activation

จากนั้นจะคำนวณ activation ของ layer ปัจจุบัน:

\[
a^{(l)} = f(z^{(l)})
\]

โดย \( f(x) \) จะเป็น activation function เช่น ReLU, Sigmoid, หรือ Tanh

---

#### 1.2.3 การคำนวณแบบวนซ้ำทุก Layer

คำนวณวนซ้ำไปเรื่อยๆ ตั้งแต่

\[
a^{(0)} = \text{input vector}
\]

จนถึง \( a^{(L)} \) หรือ \( y_i \) ซึ่งเป็นการทำนายคำตอบของโมเดล

---

#### 1.2.4 การคำนวณค่าความผิดพลาด (Loss)

โดยปกติจะใช้ **Mean Square Error (MSE)** ในการคำนวณความผิดพลาด โดยเราจะทำการ forward สำหรับ input จำนวน \( N \) ตัวใน batch และ error จะถูกเฉลี่ยเพื่อให้ได้ค่าความผิดพลาดแบบสเกลลาร์

---

## 1.2 Gradient ด้วยอนุพันธ์อันดับที่ 1

### 1.2.1 การปรับค่าพารามิเตอร์

แนวคิดคือใช้ความชันของฟังก์ชัน (อนุพันธ์อันดับที่ 1) เพื่อกำหนดทิศทางในการปรับพารามิเตอร์

ตัวอย่าง:  
หากความชันระหว่าง loss \( L \) กับพารามิเตอร์ \( W^{(l)} \) มีค่า 2 หมายความว่า:

> หากเพิ่ม \( W^{(l)} \) ขึ้น 1 หน่วย ค่า loss จะเพิ่มขึ้น 2 หน่วย

ดังนั้นเพื่อให้ loss ลดลง เราจึงปรับ \( W^{(l)} \) ในทิศทางลบของความชัน และใช้อัตราการเรียนรู้ (learning rate, \( \eta \)) เพื่อกำหนดปริมาณการปรับ

---

### 1.2.2 การหาอนุพันธ์อันดับที่ 1 (Backward Pass)

Backward pass คือกระบวนการคำนวณ gradient ของ loss function ตาม parameters โดยใช้ **Chain Rule**

จะคำนวณอนุพันธ์อันดับที่ 1 ของทุก parameter โดยเริ่มจาก layer สุดท้าย \( L \) ไปจนถึง layer ที่ 0

---

#### 1.2.2.1 Compute derivative of loss w.r.t activation

\[
\delta^{(L)} = \frac{\partial L}{\partial a^{(L)}}
\]

---

#### 1.2.2.3 Gradient w.r.t weights using chain rule

\[
\frac{\partial L}{\partial W^{(l)}} = (a^{(l-1)})^T \delta^{(l)}
\]

---

#### 1.2.2.4 Backpropagate delta to previous layer

\[
\delta^{(l-1)} = \delta^{(l)} (W^{(l)})^T \odot f'(z^{(l-1)})
\]

---

# 2. หลักการทำงานของ Hessian ใน Deep Learning

## 2.1 มิติและที่มาของ Hessian

- **Gradient** (อนุพันธ์อันดับหนึ่ง):  
  \[
  \nabla L(\theta) \in \mathbb{R}^p
  \]  
  ใช้บอก slope ของ \( L \)

- **Hessian** (อนุพันธ์อันดับสอง):  
  \[
  \nabla^2 L(\theta) = H(\theta) \in \mathbb{R}^{p \times p}
  \]  
  ใช้บอก curvature ของ \( L \)

โดย \( \theta \) หมายถึงพารามิเตอร์ทั้งหมด รวมทั้ง W และ b

---

## 2.2 องค์ประกอบและสมบัติพื้นฐานของ Hessian

### 2.2.1 ทฤษฎีบทของแคลโรต์ (Clairaut's Theorem)

ถ้าอนุพันธ์ย่อยอันดับสองแบบผสมมีความต่อเนื่องบนบริเวณหนึ่ง  
จะได้ว่า:

\[
H*{i,j} = H*{j,i}
\]

ดังนั้น Hessian จะเป็น **เมทริกซ์สมมาตร (symmetric matrix)**

---

### 2.2.2 ทฤษฎีบทสเปกตรัม (Spectral Theorem)

เมื่อ Hessian \( H(\theta) \) เป็นเมทริกซ์สมมาตรจริง จะสามารถแยกสลายได้เป็น:

\[
H = Q \Lambda Q^T
\]

โดย:

#### 2.2.2.1 Λ (Lambda)

- เป็น diagonal matrix
- มี eigenvalues \( \lambda_1, \lambda_2, ..., \lambda_n \) ของ \( H \)
- ค่า eigenvalues ทั้งหมดเป็นจำนวนจริง

#### Q (Orthogonal Matrix)

- แต่ละ column คือ eigenvector
- orthonormal basis ของ \( \mathbb{R}^n \)
- \( Q^T = Q^{-1} \)

---

---

## Application Example: Deep Autoencoder

ตัวอย่างการใช้ Hessian-Free Optimization (HF)

### ขั้นตอนสำคัญ

1. ตั้งสมการเชิงเส้น:
   \[
   H p = -\nabla L
   \]
2. ใช้ Conjugate Gradient (CG) เพื่อแก้ โดยไม่ต้องสร้าง H
3. ใช้ Hessian-vector product \(Hv\)
4. อัปเดตพารามิเตอร์ด้วยทิศทาง \(p\)

ผลลัพธ์: ลู่เข้าสู่คำตอบได้รวดเร็วกว่า SGD โดยเฉพาะในพื้นที่ที่ ill-conditioned

---

## Discussion and Analysis

### ข้อดี

- ทำงานได้ดีในพื้นที่โค้งซับซ้อน
- ใช้จำนวน epoch น้อยลง

### ข้อจำกัด

- H อาจไม่เป็น positive-definite
- คำนวณหนักกว่า first-order
- ต้องจูนพารามิเตอร์ CG

---

## Conclusion

Hessian matrix ช่วยให้เข้าใจ curvature ของ loss landscape และช่วยเพิ่มประสิทธิภาพในการฝึกโมเดลผ่าน second-order methods แม้โมเดลจะมีขนาดใหญ่ แต่เทคนิคเช่น Hessian-vector product, FIM, และ K-FAC ทำให้ใช้งานได้จริงในทางปฏิบัติ

---

## References

- Frans, K. (2023). _Second-Order Optimization_. Notes on Deep Learning.
- KMUTT. (n.d.). _MTH 234 Course Material_.
- Martens, J. (2010). _Deep learning via Hessian-free optimization_.

```

```
