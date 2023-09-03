import javax.swing.plaf.nimbus.State;
import java.sql.*;
import java.util.*;

class Database {
   public static void main(String[] args)throws Exception{
      DriverManager.registerDriver(new oracle.jdbc.driver.OracleDriver());
//      Connection con = DriverManager.getConnection("jdbc:oracle:thin:@sabzevi2.homeip.net:1521:orcl", "csus", "student");
      Statement st = con.createStatement();
      Statement st2 = con.createStatement();
      Scanner in = new Scanner(System.in);
      try {
         dropTables(st);  // need to do a try/except block for when the table doesn't exist
      } catch (SQLException e){
         System.out.println("Error in main() (tables don't exist) " + e);
      }

      createTables(st);
      fillTables(st);
      while (true) {
         getCustomerData(st);
         getOrderData(st);
         String action = getInput("Would you like to insert (i), update (u), delete (d), view (v), or exit program (quit): ", in, Arrays.asList("i", "u", "d", "v", "quit"));
         if (action.equals("i")) {
            insert(st, con, in);
         }
         if (action.equals("u")) {
            update(st, con, in);
         }
         if (action.equals("d")) {
            delete(st, st2, con, in);
         }
         if (action.equals("v")) {
            view(st);
         }
         if (action.equals("quit")) {
            quit();
         }
      }
   }

   /**
    * Display records from child table and corresponding info from parent table
    * Ex: product | customer_name | email
    */
   public static void view(Statement st) {
      try {
         ResultSet results = st.executeQuery("SELECT ughh_order.product, ughh_customer.customer_name, ughh_customer.email FROM ughh_order INNER JOIN ughh_customer ON ughh_order.customer_id = ughh_customer.customer_id");
         System.out.println("Displaying: product | customer_name | email");
         while (results.next()) {
            System.out.println(results.getString(1) + " | " + results.getString(2) + " | " + results.getString(3));
         }
         System.out.println();
      } catch (SQLException e) {
         System.out.println("Error in view(): " + e);
      }
   }

   /**
    * Asks for 'customer_id', the column to update (name or email), then the new data
    */
   public static void update(Statement st, Connection con, Scanner in) {
      String customerID = getInput("Enter CustomerID to update: ", in, getAvailableIDs(st, "customer"));
      String updateColumn = getInput("Update name (n) or email (e)?: ", in, Arrays.asList("n", "e"));
      String newData = getInput("Enter new info: ", in, Arrays.asList());
      try {
         String sql = "";
         if (updateColumn.equals("n")) {  // use 'customer_name' column
            sql = "UPDATE ughh_customer SET customer_name = ? WHERE customer_id = ?";
         }
         if (updateColumn.equals("e")) {  // use 'email' column
            sql = "UPDATE ughh_customer SET email = ? WHERE customer_id = ?";
         }
         PreparedStatement prepared = con.prepareStatement(sql);
         prepared.setString(1, newData);  // 1st '?' in sql string
         prepared.setString(2, customerID);  // 2nd '?' in sql string
         prepared.addBatch();
         prepared.executeBatch();
      } catch (SQLException e) {
         System.out.println("Error in update(): " + e);
      }
   }

   /**
    * Asks for 'order_id', deletes that row.
    * @param st/st2: requires 2 statements cuz bug.
    */
   public static void delete(Statement st, Statement st2, Connection con, Scanner in) {
      String orderID = getInput("Enter OrderID to delete: ", in, getAvailableIDs(st, "order"));
      try {
         ResultSet getCustomerID = st2.executeQuery("SELECT customer_id FROM ughh_order WHERE order_id = " + orderID);
         String customerID = "";
         while (getCustomerID.next()) {
            customerID = getCustomerID.getString(1);
         }
         st.executeQuery("DELETE FROM ughh_order WHERE order_id = " + orderID);
         System.out.println("OrderID " + orderID + " has been deleted.");
         hasForeignKeyInstances(st, st2, con, customerID);  // delete the customer if they have no other orders
      } catch (SQLException e) {
         System.out.println("Error in delete(): " + e);
      }
   }

   // checks if customer_id is used anywhere

   /**
    * Checks if a customer ID is associated with any orders, if not, it deletes them.
    * Helper function for delete()
    * @param customerID: The customer ID from the deleted order ID
    */
   public static void hasForeignKeyInstances(Statement st, Statement st2, Connection con, String customerID) {
      try {
         ResultSet getIDCount = st2.executeQuery("SELECT COUNT(*) FROM ughh_order WHERE customer_id = " + customerID);
         if (getIDCount.next()) {
            int check_zero = Integer.parseInt(getIDCount.getString(1));
            if (check_zero == 0) {
               System.out.println("Customer ID " + customerID + " deleted since no foreign key instances");
               st2.executeQuery("DELETE FROM ughh_customer WHERE customer_id = " + customerID);
            }
         }
      } catch (SQLException e) {
         System.out.println("Error in hasForeignKeyInstances(): " + e);
      }
   }

   // asks for data to insert
   public static void insert(Statement st, Connection con, Scanner in) {
      String table = getInput("Insert into customer (c) or order (o) table?: ", in, Arrays.asList("c", "o"));
      if (table.equals("c")) {
         System.out.println("Customer table selected.");
         insertSQL(st, con, "ughh_customer", getCustomerInput(in));
      } else if (table.equals("o")) {
         System.out.println("Order table selected.");
         insertSQL(st, con, "ughh_order", getOrderInput(st, in));
      } else {
         System.out.println("Neither table selected.");
      }
   }

   // Gets insert data for customer table
   public static String[] getCustomerInput(Scanner in) {
      String[] result = new String[2];
      result[0] = getInput("Enter customer name: ", in, Arrays.asList());
      result[1] = getInput("Enter customer email: ", in, Arrays.asList());
      return result;
   }

   // Gets insert data for order table
   public static String[] getOrderInput(Statement st, Scanner in) {
      String[] result = new String[2];
      result[0] = getInput("Enter product name: ", in, Arrays.asList());
      result[1] = getInput("Enter customer id: ", in, getAvailableIDs(st, "customer"));
      return result;
   }

   /**
    * Actually inserts the data.
    * Helper function for insert()
    * @param table: 'ughh_customer' or 'ughh_order'
    * @param values: the values to insert to the table
    */
   public static void insertSQL(Statement st, Connection con, String table, String[] values) {
      try {
         String sql = "";
         if (table.equals("ughh_customer")){
            sql = "SELECT customer_id FROM ughh_customer";
         }
         if (table.equals("ughh_order")){
            sql = "SELECT order_id FROM ughh_order";
         }
         ResultSet results = st.executeQuery(sql);
         int pk = 0;
         while (results.next()) {  // the last result will be the last id returned, which +1 will be used as pk
            pk = results.getInt(1) + 1;
         }
         sql = "INSERT INTO " + table + " VALUES (?, ?, ?)";
         PreparedStatement prepared = con.prepareStatement(sql);
         prepared.setString(1, Integer.toString(pk));
         prepared.setString(2, values[0]);
         prepared.setString(3, values[1]);
         prepared.addBatch();
         prepared.executeBatch();
      } catch (SQLException e) {
         System.out.println("Error in insertSQL(): " + e);
      }
   }

   // exists program
   public static void quit() {
//      List<String> messages = Arrays.asList("y", "n");
//      String input = getInput("Enter yes (y) or no (n) to exit program: ", in, messages);
//      if (input.equals("y")) {
         System.out.println("Quitting program.");
         System.exit(0);
//      }
   }

   // create customer and order tables
   public static void createTables(Statement st) throws SQLException {
      st.executeQuery("create table ughh_customer (customer_id varchar(10) PRIMARY KEY, customer_name varchar(50), email varchar(50))");
      st.executeQuery("create table ughh_order (order_id varchar(10) PRIMARY KEY, product varchar(50), customer_id varchar(50), FOREIGN KEY (customer_id) REFERENCES ughh_customer(customer_id))");
   }

   // drop customer and order tables
   public static void dropTables(Statement st) throws SQLException {
      st.executeQuery("drop table ughh_order");
      st.executeQuery("drop table ughh_customer");
   }

   // prefill customer and order tables
   public static void fillTables(Statement st) throws SQLException {
      st.executeQuery("insert into ughh_customer values (1,'name1', 'email1')");
      st.executeQuery("insert into ughh_customer values (2,'name2', 'email2')");
      st.executeQuery("insert into ughh_customer values (3,'name3', 'email3')");
      st.executeQuery("insert into ughh_order values (1,'product1', '1')");
      st.executeQuery("insert into ughh_order values (2,'product2', '1')");
      st.executeQuery("insert into ughh_order values (3,'product3', '1')");
      st.executeQuery("insert into ughh_order values (4,'product4', '2')");
      st.executeQuery("insert into ughh_order values (5,'product5', '3')");
   }

   // for testing - gets all customer table data
   public static void getCustomerData(Statement st) throws SQLException {
      ResultSet results = st.executeQuery("select * from ughh_customer ");
      System.out.println("Customer data: ");
      while (results.next()) {
         System.out.println(results.getString(1) + " | " + results.getString(2) + " | " + results.getString(3));
      }
   }

   // for testing - gets all order table data
   public static void getOrderData(Statement st) throws SQLException {
      ResultSet results = st.executeQuery("select * from ughh_order ");
      System.out.println("Order data: ");
      while (results.next()) {
         System.out.println(results.getString(1) + " | " + results.getString(2) + " | " + results.getString(3));
      }
   }

   /**
    * Gets the available IDs of a table, helps with checking if an inputted ID exists
    * @param table: 'ughh_customer' or 'ughh_order'
    * @return: a string list of IDs, or an empty string list otherwise
    */
   public static List<String> getAvailableIDs(Statement st, String table) {
      List<String> availableIDs = new ArrayList<>();
      try {
         if (table.equals("customer")) {
            ResultSet result = st.executeQuery("SELECT customer_id FROM ughh_customer");
            while (result.next()) {
//               String theID = ;
               availableIDs.add(result.getString(1));
            }
            return availableIDs;
         }
         if (table.equals("order")) {
            ResultSet result = st.executeQuery("SELECT * FROM ughh_order");
            while (result.next()) {
//               String theID = result.getString(1);
               availableIDs.add(result.getString(1));
            }
            return availableIDs;
         }
      } catch (SQLException e) {
         System.out.println("Error in checkIDExists(): " + e);
      }
      return availableIDs;
   }

   // get user input, 'expected_input' is a string list of allowed inputs
   public static String getInput(String question, Scanner in, List<String> expected_input) {
      System.out.println(question);
      String input = in.nextLine();
      if (expected_input.isEmpty()) {  // no expected input
         return input;
      } else {  // expected input
         if (expected_input.contains(input)) {  // if input == expected input
            return input;
         } else {  // if input != expected input
            while (!expected_input.contains(input)) {  // while the user input != expected input, keep asking
               System.out.println("Incorrect input... " + question);
               input = in.nextLine();
            }
            return input;  // input == expected input
         }
      }
   }


}

