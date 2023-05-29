Update of Candy Alpha 11
===

 - Updated 'share' keyword
 ```
 [at global]
 name = "k" #global: name="k"
 [at global/function:createUser]
 name = "j" #global: name="k"  local: name="j"
 share name #global: name="j"  local: name="j"
 name = "f" #global: name="k"  local: name="f"
 ```
 - Updated 'forget' keyword
 - Now, don't need to use '$' for variable. (auto detecting)
 ```
 set name to "Asdf"
 function run:
    println ("Hello, world!")
 println (name)
 ```
 -> "Asdf"
 ```
 println (run)
 ```
 -> "Hello, world!"
 -> None
 ```
 println ($run)
 ```
 -> <function:run>