## R debugging tutorial

# the newFunc
newFunc <- function(string) {
  print(aNonExistingVariable)
}

myFunc <- function(a, b) {
  browser()
  c = 2 * a
  d = 5 * b
  return(2*c + newFunc(d))
}

## call the myFuns
myFunc(2,3)

## Method 1: traceback()
traceback()

## Method 2: debug(myFunc) with multiple levels of functions
debug(myFunc)
# error found from line to line
# to exit, input undebug()
undebug(myFunc)

## Method 3: browser()
# myFunc <- function(a, b) {
#   browser() ##
#   c = 2 * a
#   d = 5 * b
#   return(2*c + newFunc(d))
# }

## use the ls() to check for variables
# Browse[1]>
#   debug at #3: c = 2*a
# Browse[2]>
#   debug at #4: d = 5*b
# Browse[2]> a
# [1] 2
# Browse[2]> b
# [1] 5
# Browse[2]> c
# Error in print(aNonexistingVariable):
#   object 'aNonExistingVariable' not found