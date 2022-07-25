<h1>0x00. Linear Algebra</h1>
<div><br></div>
<ul>
    <li>&nbsp;By Alexa Orrico, Software Engineer at Holberton School</li>
    <li>&nbsp;Weight: 2</li>
    <li>&nbsp;Ongoing project - started&nbsp;<div><span title="">Jul 25, 2022</span></div>, must end by&nbsp;<div><span title="">Jul 29, 2022</span></div>&nbsp;- you&apos;re done with&nbsp;0% of tasks.</li>
    <li>&nbsp;Checker will be released at&nbsp;<div><span title="">Jul 27, 2022 12:00 AM</span></div>
    </li>
    <li>&nbsp;An auto review will be launched at the deadline</li>
</ul>
<div>
    <div>
        <p><img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/54daaf81421a9b894688.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOU5BHMTQX4%2F20220725%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220725T140656Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=25a9d6f950c20e168e29f9b5515c5ebc22dba8b6a442360c2e135f855d262948" alt=""></p>
        <h2>Resources</h2>
        <p><strong>Read or watch</strong>:</p>
        <ul>
            <li><a href="https://intranet.hbtn.io/rltoken/C05mTOfKzZgz_AVSosNKIw" target="_blank" title="Introduction to vectors">Introduction to vectors</a></li>
            <li><a href="https://intranet.hbtn.io/rltoken/vLe4BBPfmLXy2s_Idqo87w" target="_blank" title="What is a matrix?">What is a matrix?</a> (<em>not&nbsp;<a href="https://intranet.hbtn.io/rltoken/Zad2ReJ9SU4IuQ3ZX986qw" target="_blank" title="the matrix">the matrix</a></em>)</li>
            <li><a href="https://intranet.hbtn.io/rltoken/xHWwQjqH9tgEcskvFQaV7A" target="_blank" title="Transpose">Transpose</a></li>
            <li><a href="https://intranet.hbtn.io/rltoken/2tYcOFY35stXjd0nhTpgFA" target="_blank" title="Understanding the dot product">Understanding the dot product</a></li>
            <li><a href="https://intranet.hbtn.io/rltoken/pV4znghCxaXAAny4Ou-cNw" target="_blank" title="Matrix Multiplication">Matrix Multiplication</a></li>
            <li><a href="https://intranet.hbtn.io/rltoken/ih50DhE4FvilyItYPo1x5A" target="_blank" title="What is the relationship between matrix multiplication and the dot product?">What is the relationship between matrix multiplication and the dot product?</a></li>
            <li><a href="https://intranet.hbtn.io/rltoken/DnAvjbmojZutluWV9OJVOg" target="_blank" title="The Dot Product, Matrix Multiplication, and the Magic of Orthogonal Matrices">The Dot Product, Matrix Multiplication, and the Magic of Orthogonal Matrices</a> (<em>advanced</em>)</li>
            <li><a href="https://intranet.hbtn.io/rltoken/MBHHb0eiN0OummbEdI9g_Q" target="_blank" title="numpy tutorial">numpy tutorial</a> (<em>until Shape Manipulation (excluded)</em>)</li>
            <li><a href="https://intranet.hbtn.io/rltoken/L8RdIDGi3GGO-_erGcMORg" target="_blank" title="numpy basics">numpy basics</a> (<em>until Universal Functions (included)</em>)</li>
            <li><a href="https://intranet.hbtn.io/rltoken/1LPk4EosRetS_C7eX-mQNA" target="_blank" title="array indexing">array indexing</a></li>
            <li><a href="https://intranet.hbtn.io/rltoken/slRzAgt6aom5-Nj5XSdUcQ" target="_blank" title="numerical operations on arrays">numerical operations on arrays</a></li>
            <li><a href="https://intranet.hbtn.io/rltoken/xgq6QIOHufhg8lHCZn0jwA" target="_blank" title="Broadcasting">Broadcasting</a></li>
            <li><a href="https://intranet.hbtn.io/rltoken/Woz5KooXnb7GhIFuLS-Ndw" target="_blank" title="numpy mutations and broadcasting">numpy mutations and broadcasting</a></li>
        </ul>
        <p><strong>References</strong>:</p>
        <ul>
            <li><a href="https://intranet.hbtn.io/rltoken/Ah-QtZhAhFSYnloj837a8Q" target="_blank" title="numpy.ndarray">numpy.ndarray</a></li>
            <li><a href="https://intranet.hbtn.io/rltoken/mvx-STJbJ4Nn1N_BFfpnaQ" target="_blank" title="numpy.ndarray.shape">numpy.ndarray.shape</a></li>
            <li><a href="https://intranet.hbtn.io/rltoken/I1V8iDWar7Hnoh_VwQzZ_Q" target="_blank" title="numpy.transpose">numpy.transpose</a></li>
            <li><a href="https://intranet.hbtn.io/rltoken/iv73fN04gTbpeV_XcIIaPQ" target="_blank" title="numpy.ndarray.transpose">numpy.ndarray.transpose</a></li>
            <li><a href="https://intranet.hbtn.io/rltoken/MbHJEqjwavimnL8HRtaYCA" target="_blank" title="numpy.matmul">numpy.matmul</a></li>
        </ul>
        <h2>Learning Objectives</h2>
        <p>At the end of this project, you are expected to be able to&nbsp;<a href="https://intranet.hbtn.io/rltoken/HXMfblwaZlByv5YItZHtGA" target="_blank" title="explain to anyone">explain to anyone</a>,&nbsp;<strong>without the help of Google</strong>:</p>
        <h3>General</h3>
        <ul>
            <li>What is a vector?</li>
            <li>What is a matrix?</li>
            <li>What is a transpose?</li>
            <li>What is the shape of a matrix?</li>
            <li>What is an axis?</li>
            <li>What is a slice?</li>
            <li>How do you slice a vector/matrix?</li>
            <li>What are element-wise operations?</li>
            <li>How do you concatenate vectors/matrices?</li>
            <li>What is the dot product?</li>
            <li>What is matrix multiplication?</li>
            <li>What is&nbsp;<code>Numpy</code>?</li>
            <li>What is parallelization and why is it important?</li>
            <li>What is broadcasting?</li>
        </ul>
        <h2>Requirements</h2>
        <h3>Python Scripts</h3>
        <ul>
            <li>Allowed editors:&nbsp;<code>vi</code>,&nbsp;<code>vim</code>,&nbsp;<code>emacs</code></li>
            <li>All your files will be interpreted/compiled on Ubuntu 20.04 LTS using&nbsp;<code>python3</code> (version 3.8)</li>
            <li>Your files will be executed with&nbsp;<code>numpy</code> (version 1.19.2)</li>
            <li>All your files should end with a new line</li>
            <li>The first line of all your files should be exactly&nbsp;<code>#!/usr/bin/env python3</code></li>
            <li>A&nbsp;<code>README.md</code> file, at the root of the folder of the project, is mandatory</li>
            <li>Your code should follow&nbsp;<code>pycodestyle</code> (version 2.6)</li>
            <li>All your modules should have documentation (<code>python3 -c &apos;print(__import__(&quot;my_module&quot;).__doc__)&apos;</code>)</li>
            <li>All your classes should have documentation (<code>python3 -c &apos;print(__import__(&quot;my_module&quot;).MyClass.__doc__)&apos;</code>)</li>
            <li>All your functions (inside and outside a class) should have documentation (<code>python3 -c &apos;print(__import__(&quot;my_module&quot;).my_function.__doc__)&apos;</code> and&nbsp;<code>python3 -c &apos;print(__import__(&quot;my_module&quot;).MyClass.my_function.__doc__)&apos;</code>)</li>
            <li><strong>Unless otherwise noted, you are not allowed to import any module</strong></li>
            <li>All your files must be executable</li>
            <li>The length of your files will be tested using&nbsp;<code>wc</code></li>
        </ul>
        <h2>More Info</h2>
        <h3>Installing Ubuntu 20.04 and Python 3.8</h3>
        <p>Follow the instructions listed in&nbsp;<code>Using Vagrant on your personal computer</code>, should be using&nbsp;<code>ubuntu/focal64</code>.</p>
        <p><em>Python 3.8 comes pre-installed on Ubuntu 20.04. How convenient! You can confirm this with</em> <code>python3 -V</code></p>
        <h3>Installing pip (latest)</h3>
        <p><a href="https://intranet.hbtn.io/rltoken/bnipr2zxol-aSqNNCglaFg" target="_blank" title="pip installation">pip installation</a></p>
        <h3>Installing numpy 1.19.2, scipy 1.6.2, and pycodestyle 2.6</h3>
        <pre><code>$ pip install --user numpy==1.19.2
$ pip install --user scipy==1.6.2
$ pip install --user pycodestyle==2.6
</code></pre>
        <p>To check that all have been successfully downloaded, use&nbsp;<code>pip list</code>.</p>
    </div>
</div>