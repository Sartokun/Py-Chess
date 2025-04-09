# ♟️ Gardner Chess AI

AI เล่นหมากรุกแบบเรียนรู้ด้วยตัวเอง (Reinforcement Learning) สร้างด้วย Python, PyTorch และ Pygame  
เรียนรู้ผ่านเทคนิค Deep Q-Learning พร้อม UI ให้ดูผลลัพธ์แบบ real-time

---

## 🔍 โปรเจกต์นี้คืออะไร?

GardnerChessAI คือ AI ที่เรียนรู้การเล่นหมากรุก **โดยไม่มีการสอนล่วงหน้า**  
ใช้แนวคิดของ **Deep Q-Learning (DQN)** เพื่อให้ AI ค่อยๆ พัฒนาแนวทางการเล่นด้วยตัวเอง

---

## 🧠 Neural Network คืออะไร?

ลองนึกถึงสมองมนุษย์ที่มีเซลล์ประสาทเชื่อมกัน — Neural Network ก็คือการจำลองไอเดียนี้ด้วยคอมพิวเตอร์

- รับ **ข้อมูลกระดานหมากรุก** เป็น input
- คำนวณผ่านหลายชั้นของ “neuron”
- ให้ output เป็น **คะแนน (Q-value)** สำหรับแต่ละการเดิน

```
Input (board) → Hidden layers → Q-value ของแต่ละการเดิน
```

---

## 🤖 แล้ว Deep Q-Learning คืออะไร?

DQN คือการ:

1. ประเมินว่า “ถ้าเดินหมากนี้จะดีแค่ไหน” → Q-value
2. ถ้า AI เดินดี (กินหมาก, ชนะ) → เพิ่มรางวัล
3. ถ้าเดินแล้วเสียเปรียบ → ลดรางวัล
4. ฝึก Neural Network ให้คาดการณ์ Q-value ได้แม่นขึ้นเรื่อยๆ

AI จะเล่นกับตัวเองซ้ำๆ แล้วค่อยๆ พัฒนาการเดินเองแบบ **Self-Play**

---

## 🛠️ เทคโนโลยีที่ใช้

- 🐍 Python 3.9+
- 💡 PyTorch (Neural Network)
- 🎮 Pygame (UI)
- ♟️ python-chess (หมากรุก)
- 🧠 NumPy (จัดการข้อมูล)

---

## 🚀 วิธีติดตั้งและใช้งาน

1. Clone โปรเจกต์นี้
```bash
git clone https://github.com/yourusername/gardnerChessAi.git
cd gardnerChessAi
```

2. ติดตั้ง dependencies
```bash
pip install -r requirements.txt
```

3. รันโปรเจกต์
```bash
python main.py
```

> จะมี UI โชว์กระดานหมากรุกและการเดินของ AI แบบสดๆ

---

## 🎮 วิธีที่ AI เรียนรู้

```python
for game_num in range(5):
    board = chess.Board()
    while not board.is_game_over():
        state = board_to_state(board)
        legal_moves = list(board.legal_moves)
        move = agent.act(state, legal_moves)
        board.push(move)
        next_state = board_to_state(board)
        reward = 1 if board.is_checkmate() else 0
        agent.remember(state, move, reward, next_state, board.is_game_over())
        agent.replay()
```

AI จะ:

- สุ่มเดินตอนเริ่มต้น
- ประเมินผลจากการเล่น
- จำว่าการเดินไหน “คุ้มค่า”
- ค่อยๆ เลือกการเดินฉลาดขึ้นเรื่อยๆ

---

## 📊 UI ที่แสดงผล

- กระดานหมากรุกแบบเรียลไทม์
- แถบด้านข้างแสดง Move History และ Q-value
- สถิติต่างๆ เช่น จำนวนเกมที่ชนะ, ความแม่นยำของ Q-value ฯลฯ

---

## 🧪 ตัวอย่างการใช้งาน

เมื่อรัน `main.py` จะเห็นกระดานที่ AI เล่นกับตัวเอง พร้อมกราฟิกและสถิติแบบเรียลไทม์

![screenshot-placeholder](https://via.placeholder.com/480x300?text=Chess+UI+Preview)

---

## 📚 อ้างอิง

- [Gardner Chess AI (GitHub)](https://github.com/flowun/gardnerChessAi)
- [AlphaZero Concept - Chess.com](https://www.chess.com/terms/alphazero-chess-engine)

---

## 🙌 ฝากไว้

ถ้าคุณชอบโปรเจกต์นี้ อย่าลืม ⭐️ ให้กำลังใจ หรือลองนำไปต่อยอดนะครับ 😄