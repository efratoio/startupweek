import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import axios from 'axios';
import GraphComp from './components/graph';
class App extends Component {

  get_curr_monthly(){
    
    axios.get(`${process.env.REACT_APP_USERS_SERVICE_URL}/graph`)
    .then((res) => { console.log(res); })
    .catch((err) => { console.log(err); })
  }
  
  render() {
    this.get_curr_monthly();
    return (
      <div className="App">
        <header className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <h1 className="App-title">Welcome to React</h1>
        </header>
        <div>
      <GraphComp data={[5,10,1,3]} size={[500,500]} />
      </div>
      </div>
    );
  }
}

export default App;
