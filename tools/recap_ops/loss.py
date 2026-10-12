import re, io, statistics as st
txt = io.open('/workspace/a100b_ep1.log', errors='ignore').read()
rows=[]
for m in re.finditer(r'step=([0-9.]+).*?train/learning_rate=([0-9.eE+-]+).*?train/loss=([0-9.eE+-]+)', txt):
    rows.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
# the logged "step" is fractional (epoch-ish); use index order
print(f'logged points: {len(rows)}')
loss=[r[2] for r in rows]; lr=[r[1] for r in rows]
def w(a,b,label):
    seg=loss[a:b]
    print(f'  {label:22s} n={len(seg):4d}  mean={st.mean(seg):.5f}  lr_mean={st.mean(lr[a:b]):.3e}')
n=len(loss)
print('\n=== windowed means ===')
w(0,40,'first 40');  w(40,400,'40-400'); w(400,800,'400-800')
w(n//2-200,n//2+200,'middle 400')
w(n-800,n-400,'last 800..400'); w(n-400,n-200,'last 400..200'); w(n-200,n,'last 200')
print('\n=== last 600, split into three 200s (the plateau question) ===')
w(n-600,n-400,'[-600,-400)'); w(n-400,n-200,'[-400,-200)'); w(n-200,n,'[-200,0)')
print('\n=== learning rate at the end ===')
for i in (n-1200, n-600, n-300, n-100, n-1):
    print(f'  idx {i:5d}  lr={lr[i]:.3e}  loss={loss[i]:.5f}')
print('\n=== cumulative mean differential over the last 1000 ===')
import itertools
for k in (1000,800,600,400,200,100):
    a=st.mean(loss[n-2*k:n-k]); b=st.mean(loss[n-k:n])
    print(f'  prev {k:4d} mean {a:.5f}  ->  last {k:4d} mean {b:.5f}   delta {b-a:+.6f} ({100*(b-a)/a:+.2f}%)')
