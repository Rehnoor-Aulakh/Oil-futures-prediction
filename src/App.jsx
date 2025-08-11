// App.jsx
import Block from "./Block";
import Header from "./Header";

function App() {
  return (
    <div className="bg-contain min-h-screen flex flex-col mt-14">
      <Header />
      <div className="flex justify-center">
        <Block />
      </div>
    </div>
  );
}

export default App;
