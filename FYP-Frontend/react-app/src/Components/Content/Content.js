import { useState } from 'react';
import './Content.css'
import CircularIndeterminate from '../CircularBar/CircularBar';
function Content() {

   const [inputVal,setInputVal]=useState('')
   const [isLoading,setIsLoading]=useState(false)
   const [outputVal,setOutputVal]=useState('Predicted Python Code ...')

    const buttonClickHandler = ( )=>{
      setIsLoading(true)
      fetch("http://127.0.0.1:5000/convert", {
      method: "POST",
      body: JSON.stringify({str:inputVal}),
      headers: { "content-type": "application/json" },
    })
      .then((res) => {
        console.log('here')
        if (!res.ok) return Promise.reject(res);
        console.log('pending')
        return res.json();
      })
      .then((data) => {
       setIsLoading(false)
        console.log(data)
        setOutputVal(data.response)
      })
      .catch(console.error);
    }

    const handleChange=(event)=>{
      setInputVal(event.target.value)
    }

    return ( 
        <div className="content-div">
           <div className="center-div">
            <input onChange={handleChange} className='input-style' placeholder="Enter String here" value={inputVal}/>
           </div>

           <div className='center-div'>
            <button className={inputVal==='' ? 'button-style-disable' : 'button-style'} disabled={inputVal===''} onClick={buttonClickHandler}>Generate Code</button>
           </div>

           <div className='center-div'>
            <div className='output-style'>
               {!isLoading && <pre>{outputVal}</pre>}
               {isLoading && <>
                  <pre>Wait Code is being generated ... </pre>
                  <div className='circular-bar'>
                     <CircularIndeterminate/>
                  </div>
               </>}
            </div>
           </div>
        </div>
       
     );
}

export default Content;